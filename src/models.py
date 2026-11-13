"""
Model definitions and evaluation metrics for Cinema Ticket Demand Forecasting.
Includes Naive Capacity-Adjusted Baseline, Ridge Regression, and Gradient Boosted Regressor.
"""
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from src.features import create_preprocessor, add_engineered_features, RAW_FEATURE_COLS


class CapacityAdjustedBaseline(BaseEstimator, RegressorMixin):
    """
    Naive baseline using historical average occupancy grouped by
    (genre, is_weekend, time_slot) scaled by screen capacity.
    """
    def __init__(self):
        self.group_occupancy_lookup = {}
        self.global_mean_occupancy = 0.50

    def fit(self, X: pd.DataFrame, y: pd.Series):
        df = X.copy()
        df["target_tickets"] = y.values
        df["observed_occupancy"] = df["target_tickets"] / df["screen_capacity"].clip(lower=1)
        
        self.global_mean_occupancy = float(df["observed_occupancy"].mean())
        
        # Grouped lookup
        grouped = df.groupby(["genre", "is_weekend", "time_slot"])["observed_occupancy"].mean()
        self.group_occupancy_lookup = grouped.to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = []
        for _, row in X.iterrows():
            key = (row["genre"], int(row["is_weekend"]), row["time_slot"])
            occupancy = self.group_occupancy_lookup.get(key, self.global_mean_occupancy)
            pred_tickets = occupancy * row["screen_capacity"]
            # Bounded by physical screen capacity
            pred_bounded = np.clip(pred_tickets, 0, row["screen_capacity"])
            predictions.append(round(pred_bounded))
        return np.array(predictions, dtype=int)


class CapacityBoundedRegressor(BaseEstimator, RegressorMixin):
    """
    Ensemble regressor with strict physical capacity capping and
    residual-based uncertainty intervals.
    """
    def __init__(self, base_estimator=None, alpha: float = 1.0):
        self.alpha = alpha
        self.base_estimator = base_estimator or HistGradientBoostingRegressor(
            random_state=42,
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=20,
        )
        self.pipeline = None
        self.residual_std_ = 10.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        preprocessor = create_preprocessor()
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", self.base_estimator),
        ])
        
        X_eng = add_engineered_features(X)
        self.pipeline.fit(X_eng, y)
        
        # Compute residual standard deviation on training set for prediction intervals
        raw_preds = self.pipeline.predict(X_eng)
        bounded_preds = np.clip(raw_preds, 0, X["screen_capacity"].values)
        residuals = y.values - bounded_preds
        self.residual_std_ = float(np.std(residuals))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_eng = add_engineered_features(X)
        raw_preds = self.pipeline.predict(X_eng)
        # Physical constraint: tickets sold cannot be negative or exceed screen capacity
        capacities = X["screen_capacity"].values
        bounded_preds = np.clip(raw_preds, 0, capacities)
        return np.round(bounded_preds).astype(int)

    def predict_with_intervals(self, X: pd.DataFrame, confidence: float = 0.80) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (predicted_tickets, lower_bound, upper_bound)."""
        preds = self.predict(X)
        z = 1.282 if confidence == 0.80 else 1.96  # 80% or 95%
        margin = z * self.residual_std_
        
        capacities = X["screen_capacity"].values
        lower_bounds = np.clip(np.round(preds - margin), 0, capacities).astype(int)
        upper_bounds = np.clip(np.round(preds + margin), 0, capacities).astype(int)
        return preds, lower_bounds, upper_bounds


def compute_forecasting_metrics(y_true: np.ndarray, y_pred: np.ndarray, capacities: np.ndarray) -> Dict[str, float]:
    """Compute standard forecasting regression and cinema business metrics."""
    abs_errors = np.abs(y_true - y_pred)
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    total_actual = np.sum(y_true)
    wape = float(np.sum(abs_errors) / total_actual) if total_actual > 0 else 0.0
    
    # Occupancy rate MAE
    occupancy_true = y_true / capacities
    occupancy_pred = y_pred / capacities
    occupancy_mae = float(np.mean(np.abs(occupancy_true - occupancy_pred)))
    
    # Capacity overrun check (% of raw predictions that would exceed physical capacity)
    overrun_count = int(np.sum(y_pred > capacities))
    overrun_pct = float(overrun_count / len(y_pred) * 100)
    
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "WAPE_pct": round(wape * 100, 2),
        "Occupancy_MAE_pct": round(occupancy_mae * 100, 2),
        "Capacity_Overrun_pct": round(overrun_pct, 2),
    }
