"""
Training and Evaluation Pipeline for Cinema Ticket Demand Forecasting.
Executes chronological split, benchmarks baseline vs Ridge vs Gradient Boosting,
and saves the trained artifact with evaluation metadata.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from src.config import (
    DATA_PATH,
    MODEL_PATH,
    METADATA_PATH,
    FEATURE_NAMES_PATH,
    RANDOM_SEED,
)
from src.features import (
    RAW_FEATURE_COLS,
    TARGET_COL,
    chronological_train_val_test_split,
)
from src.models import (
    CapacityAdjustedBaseline,
    CapacityBoundedRegressor,
    compute_forecasting_metrics,
)


def train_and_evaluate():
    print(f"Loading cinema dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Total shows loaded: {len(df)}")
    
    # Chronological Split
    train_df, val_df, test_df = chronological_train_val_test_split(df, 0.70, 0.15)
    print(f"Chronological Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Train dates: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Val dates:   {val_df['date'].min()} to {val_df['date'].max()}")
    print(f"Test dates:  {test_df['date'].min()} to {test_df['date'].max()}")
    
    X_train = train_df[RAW_FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    
    X_val = val_df[RAW_FEATURE_COLS]
    y_val = val_df[TARGET_COL]
    val_caps = val_df["screen_capacity"].values
    
    X_test = test_df[RAW_FEATURE_COLS]
    y_test = test_df[TARGET_COL]
    test_caps = test_df["screen_capacity"].values
    
    models = {}
    
    # 1. Baseline Model
    print("\n--- Training Model 0: Capacity-Adjusted Historical Baseline ---")
    baseline = CapacityAdjustedBaseline()
    baseline.fit(X_train, y_train)
    models["CapacityAdjustedBaseline"] = baseline
    
    # 2. Model 1: Capacity-Bounded Ridge Regression
    print("--- Training Model 1: Capacity-Bounded Ridge Regression ---")
    ridge_model = CapacityBoundedRegressor(
        base_estimator=Ridge(alpha=10.0, random_state=RANDOM_SEED)
    )
    ridge_model.fit(X_train, y_train)
    models["RidgeRegression"] = ridge_model
    
    # 3. Model 2: Capacity-Bounded Gradient Boosting Regressor
    print("--- Training Model 2: Capacity-Bounded Gradient Boosting Regressor ---")
    gb_model = CapacityBoundedRegressor(
        base_estimator=HistGradientBoostingRegressor(
            random_state=RANDOM_SEED,
            max_iter=160,
            learning_rate=0.07,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.5,
        )
    )
    gb_model.fit(X_train, y_train)
    models["GradientBoosting"] = gb_model
    
    # Evaluate All Models
    eval_results = {}
    print("\n================ EVALUATION SUMMARY (TEST SET) ================")
    print(f"{'Model':<28} | {'MAE':<7} | {'RMSE':<7} | {'WAPE (%)':<9} | {'Occ MAE (%)':<12}")
    print("-" * 75)
    
    for name, model in models.items():
        val_preds = model.predict(X_val)
        val_metrics = compute_forecasting_metrics(y_val.values, val_preds, val_caps)
        
        test_preds = model.predict(X_test)
        test_metrics = compute_forecasting_metrics(y_test.values, test_preds, test_caps)
        
        # Subgroup evaluations on test set
        cold_mask = (test_df["is_cold_start"] == 1).values
        mature_mask = (test_df["is_cold_start"] == 0).values
        weekend_mask = (test_df["is_weekend"] == 1).values
        weekday_mask = (test_df["is_weekend"] == 0).values
        
        cold_mae = round(float(np.mean(np.abs(y_test.values[cold_mask] - test_preds[cold_mask]))), 2)
        mature_mae = round(float(np.mean(np.abs(y_test.values[mature_mask] - test_preds[mature_mask]))), 2)
        weekend_mae = round(float(np.mean(np.abs(y_test.values[weekend_mask] - test_preds[weekend_mask]))), 2)
        weekday_mae = round(float(np.mean(np.abs(y_test.values[weekday_mask] - test_preds[weekday_mask]))), 2)
        
        eval_results[name] = {
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "subgroup_test_metrics": {
                "cold_start_mae": cold_mae,
                "mature_release_mae": mature_mae,
                "weekend_mae": weekend_mae,
                "weekday_mae": weekday_mae,
            },
        }
        
        print(
            f"{name:<28} | {test_metrics['MAE']:<7} | {test_metrics['RMSE']:<7} | "
            f"{test_metrics['WAPE_pct']:<9} | {test_metrics['Occupancy_MAE_pct']:<12}"
        )
    print("=" * 75)
    
    # Champion selection
    champion_name = "GradientBoosting"
    champion_model = models[champion_name]
    
    # Save Model Artifact
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion_model, MODEL_PATH)
    print(f"\nChampion model ({champion_name}) saved to: {MODEL_PATH}")
    
    # Save Metadata
    metadata = {
        "model_name": champion_name,
        "algorithm": "HistGradientBoostingRegressor with Capacity Bounds",
        "random_seed": RANDOM_SEED,
        "forecast_horizon": "T-24h",
        "training_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "train_date_range": [train_df["date"].min(), train_df["date"].max()],
        "test_date_range": [test_df["date"].min(), test_df["date"].max()],
        "residual_std": champion_model.residual_std_,
        "raw_feature_columns": RAW_FEATURE_COLS,
        "model_comparison": eval_results,
    }
    
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Evaluation metadata saved to: {METADATA_PATH}")
    
    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump({"raw_features": RAW_FEATURE_COLS, "target": TARGET_COL}, f, indent=2)
    print(f"Feature specifications saved to: {FEATURE_NAMES_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
