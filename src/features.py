"""
Feature engineering and preprocessing pipeline for Cinema Ticket Demand Forecasting.
Provides strict chronological train/val/test splits and leakage-free transformations.
"""
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Categorical features in raw inputs
CATEGORICAL_FEATURES = ["genre", "time_slot", "screen_tier"]

# Raw numerical features directly available at T-24h
RAW_NUMERICAL_FEATURES = [
    "critic_rating",
    "star_power",
    "runtime_minutes",
    "days_since_release",
    "is_cold_start",
    "screen_capacity",
    "ticket_price",
    "show_hour",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "competing_releases_count",
    "prebooking_pace_t24",
    "historical_movie_avg_occupancy",
]

# Engineered numerical features derived at inference/training
ENGINEERED_FEATURES = [
    "prebook_to_capacity_ratio",
    "decay_factor",
]

ALL_TRANSFORMED_NUMERICAL = RAW_NUMERICAL_FEATURES + ENGINEERED_FEATURES
RAW_FEATURE_COLS = CATEGORICAL_FEATURES + RAW_NUMERICAL_FEATURES
TARGET_COL = "tickets_sold"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive non-leaking domain features available at T-24h."""
    df_out = df.copy()
    
    # Pre-booking saturation ratio relative to screen capacity
    df_out["prebook_to_capacity_ratio"] = (
        df_out["prebooking_pace_t24"] / df_out["screen_capacity"].clip(lower=1)
    ).clip(0.0, 1.0)
    
    # Non-linear lifecycle decay curve
    df_out["decay_factor"] = np.exp(-0.045 * df_out["days_since_release"].clip(lower=0))
    
    return df_out


def create_preprocessor() -> ColumnTransformer:
    """Create a reproducible ColumnTransformer for scikit-learn pipelines."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                ALL_TRANSFORMED_NUMERICAL,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def chronological_train_val_test_split(
    df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data strictly chronologically to prevent temporal lookahead leakage.
    Train: [0, train_ratio]
    Validation: [train_ratio, train_ratio + val_ratio]
    Test: [train_ratio + val_ratio, 1.0]
    """
    df_sorted = df.sort_values(by="show_datetime").reset_index(drop=True)
    n = len(df_sorted)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[val_end:].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df
