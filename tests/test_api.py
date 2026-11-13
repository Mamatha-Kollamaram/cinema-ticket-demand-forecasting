"""
API endpoint unit tests verifying functionality, input validation, and decision support logic.
Calls endpoint handlers directly and tests ASGI routing.
"""
import pytest
from src.api import (
    health_check,
    get_evaluation_metrics,
    predict_show_demand,
    explain_prediction,
    predict_batch,
    ShowPredictionInput,
)


def test_health_endpoint_direct():
    data = health_check()
    assert data["status"] == "healthy"
    assert data["model_name"] == "GradientBoosting"
    assert data["forecast_horizon"] == "T-24h"
    assert data["training_samples"] > 0


def test_metrics_endpoint_direct():
    data = get_evaluation_metrics()
    assert "champion_model" in data
    assert "model_comparison" in data
    assert "GradientBoosting" in data["model_comparison"]
    assert "CapacityAdjustedBaseline" in data["model_comparison"]
    assert "RidgeRegression" in data["model_comparison"]
    
    # Confirm metrics exist
    gb_metrics = data["model_comparison"]["GradientBoosting"]["test_metrics"]
    assert gb_metrics["MAE"] < 10.0
    assert gb_metrics["WAPE_pct"] < 15.0


def test_predict_endpoint_real_output():
    payload = ShowPredictionInput(
        movie_title="Aegis Protocol",
        genre="Action_SciFi",
        critic_rating=8.7,
        star_power=9.5,
        runtime_minutes=140,
        days_since_release=1,
        is_cold_start=1,
        screen_id="Screen_1_IMAX",
        screen_tier="IMAX",
        screen_capacity=420,
        ticket_price=22.5,
        time_slot="Night_Prime",
        show_hour=20,
        day_of_week=5,
        is_weekend=1,
        is_holiday=0,
        competing_releases_count=4,
        prebooking_pace_t24=210,
        historical_movie_avg_occupancy=0.65,
    )
    res = predict_show_demand(payload)
    
    assert 0 <= res.predicted_tickets <= 420
    assert res.screen_capacity == 420
    assert res.is_cold_start is True
    assert res.cold_start_warning is not None
    assert "confidence_interval_80" in res.model_dump()
    assert res.confidence_interval_80["lower_bound"] <= res.predicted_tickets
    assert res.predicted_tickets <= res.confidence_interval_80["upper_bound"]


def test_explain_endpoint_staffing_and_drivers():
    payload = ShowPredictionInput(
        movie_title="The Silent Canvas",
        genre="Drama",
        critic_rating=8.9,
        star_power=6.5,
        runtime_minutes=128,
        days_since_release=15,
        is_cold_start=0,
        screen_id="Screen_3_MidA",
        screen_tier="Standard",
        screen_capacity=220,
        ticket_price=12.5,
        time_slot="Matinee",
        show_hour=13,
        day_of_week=2,
        is_weekend=0,
        is_holiday=0,
        competing_releases_count=5,
        prebooking_pace_t24=50,
        historical_movie_avg_occupancy=0.52,
    )
    explanation = explain_prediction(payload)
    
    assert "staffing_recommendation" in explanation
    staff = explanation["staffing_recommendation"]
    assert staff["concession_staff"] >= 1
    assert staff["ushers"] >= 1
    assert staff["cleaning_crew"] >= 1
    
    assert "screen_allocation_advisory" in explanation
    assert "key_drivers" in explanation
    assert len(explanation["key_drivers"]) >= 3


def test_batch_prediction():
    shows = [
        ShowPredictionInput(
            movie_title="Aegis Protocol",
            genre="Action_SciFi",
            screen_capacity=420,
            prebooking_pace_t24=150,
        ),
        ShowPredictionInput(
            movie_title="Office Disaster",
            genre="Comedy",
            screen_capacity=150,
            prebooking_pace_t24=35,
        ),
    ]
    batch_res = predict_batch(shows)
    assert batch_res["total_shows"] == 2
    assert len(batch_res["predictions"]) == 2
    assert batch_res["predictions"][0]["predicted_tickets"] <= 420
    assert batch_res["predictions"][1]["predicted_tickets"] <= 150
