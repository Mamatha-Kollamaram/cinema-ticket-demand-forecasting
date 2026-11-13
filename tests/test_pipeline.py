"""
Unit and integration tests for data generation, feature engineering, and model bounds.
"""
import pytest
import numpy as np
import pandas as pd
from src.config import DATA_PATH, MODEL_PATH
from src.features import (
    RAW_FEATURE_COLS,
    add_engineered_features,
    chronological_train_val_test_split,
)
from src.models import CapacityAdjustedBaseline, CapacityBoundedRegressor
import joblib


def test_dataset_exists_and_meets_volume_requirement():
    assert DATA_PATH.exists(), f"Dataset file {DATA_PATH} not found"
    df = pd.read_csv(DATA_PATH)
    assert len(df) >= 1000, f"Expected at least 1000 records, found {len(df)}"
    assert "tickets_sold" in df.columns
    assert "screen_capacity" in df.columns
    # Check physical bound
    assert (df["tickets_sold"] <= df["screen_capacity"]).all(), "Tickets sold exceeds capacity in raw data"
    assert (df["tickets_sold"] >= 0).all(), "Negative tickets sold found in raw data"


def test_chronological_split_has_zero_temporal_leakage():
    df = pd.read_csv(DATA_PATH)
    train_df, val_df, test_df = chronological_train_val_test_split(df, 0.70, 0.15)
    
    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    
    max_train_time = pd.to_datetime(train_df["show_datetime"]).max()
    min_val_time = pd.to_datetime(val_df["show_datetime"]).min()
    max_val_time = pd.to_datetime(val_df["show_datetime"]).max()
    min_test_time = pd.to_datetime(test_df["show_datetime"]).min()
    
    # Strict temporal ordering
    assert max_train_time <= min_val_time, "Temporal leakage: Train overlap with Validation"
    assert max_val_time <= min_test_time, "Temporal leakage: Validation overlap with Test"


def test_feature_engineering_bounds():
    sample_df = pd.DataFrame({
        "prebooking_pace_t24": [50, 200, 500],
        "screen_capacity": [100, 200, 400],
        "days_since_release": [0, 5, 30],
    })
    transformed = add_engineered_features(sample_df)
    assert "prebook_to_capacity_ratio" in transformed.columns
    assert "decay_factor" in transformed.columns
    
    # Ratios should be bounded between 0 and 1
    assert (transformed["prebook_to_capacity_ratio"] >= 0.0).all()
    assert (transformed["prebook_to_capacity_ratio"] <= 1.0).all()
    # Decay factor should be strictly between 0 and 1
    assert (transformed["decay_factor"] > 0.0).all()
    assert (transformed["decay_factor"] <= 1.0).all()


def test_champion_model_loaded_and_satisfies_capacity_bounds():
    assert MODEL_PATH.exists(), f"Model artifact {MODEL_PATH} not found"
    model = joblib.load(MODEL_PATH)
    
    # Test on synthetic extreme edge cases
    test_cases = pd.DataFrame([
        # Case 1: Ultra high demand blockbuster in tiny VIP hall (80 seats)
        {
            "genre": "Action_SciFi",
            "time_slot": "Night_Prime",
            "screen_tier": "VIP_Gold",
            "critic_rating": 9.9,
            "star_power": 10.0,
            "runtime_minutes": 140,
            "days_since_release": 1,
            "is_cold_start": 1,
            "screen_capacity": 80,
            "ticket_price": 45.0,
            "show_hour": 20,
            "day_of_week": 5,
            "is_weekend": 1,
            "is_holiday": 1,
            "competing_releases_count": 2,
            "prebooking_pace_t24": 78,
            "historical_movie_avg_occupancy": 0.95,
        },
        # Case 2: Zero prebooking off-peak show in huge IMAX hall (420 seats)
        {
            "genre": "Drama",
            "time_slot": "Morning",
            "screen_tier": "IMAX",
            "critic_rating": 5.0,
            "star_power": 4.0,
            "runtime_minutes": 110,
            "days_since_release": 40,
            "is_cold_start": 0,
            "screen_capacity": 420,
            "ticket_price": 12.0,
            "show_hour": 10,
            "day_of_week": 1,
            "is_weekend": 0,
            "is_holiday": 0,
            "competing_releases_count": 8,
            "prebooking_pace_t24": 5,
            "historical_movie_avg_occupancy": 0.20,
        },
    ])
    
    preds, lower, upper = model.predict_with_intervals(test_cases)
    
    # Check predictions are integers and strictly bounded
    for i in range(len(test_cases)):
        cap = test_cases.loc[i, "screen_capacity"]
        assert 0 <= preds[i] <= cap, f"Prediction {preds[i]} violated physical capacity {cap}"
        assert 0 <= lower[i] <= upper[i] <= cap, f"Intervals [{lower[i]}, {upper[i]}] violated capacity {cap}"

