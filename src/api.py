"""
FastAPI Inference Service and Interactive Decision Support Dashboard for Cinema Ticket Demand Forecasting.
Provides endpoints for single and batch predictions, explainability, cold-start diagnostics, and held-out metrics.
"""
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    MODEL_PATH,
    METADATA_PATH,
    STATIC_DIR,
    SCREEN_CONFIGS,
)
from src.features import RAW_FEATURE_COLS
from src.explain import explain_single_prediction

app = FastAPI(
    title="Cinema Ticket Demand Forecasting API",
    description="Operational decision-support ML system for cinema screen & staff allocation.",
    version="1.0.0",
)

# Enable CORS for flexible dashboard and client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Lazy-loaded model and metadata caches
_MODEL = None
_METADATA = None


def get_model():
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model artifact not found at {MODEL_PATH}. Run 'python -m src.train' first.")
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def get_metadata():
    global _METADATA
    if _METADATA is None:
        if not METADATA_PATH.exists():
            raise RuntimeError(f"Metadata not found at {METADATA_PATH}. Run 'python -m src.train' first.")
        with open(METADATA_PATH, "r") as f:
            _METADATA = json.load(f)
    return _METADATA


class ShowPredictionInput(BaseModel):
    movie_title: str = Field(default="Aegis Protocol", description="Movie title")
    genre: str = Field(default="Action_SciFi", description="Movie genre")
    critic_rating: float = Field(default=8.7, ge=1.0, le=10.0, description="Critic score 1-10")
    star_power: float = Field(default=9.5, ge=1.0, le=10.0, description="Star power index 1-10")
    runtime_minutes: int = Field(default=135, ge=60, le=240, description="Runtime in minutes")
    days_since_release: int = Field(default=1, ge=0, le=90, description="Days since theatrical release")
    is_cold_start: int = Field(default=1, ge=0, le=1, description="1 if days_since_release <= 3 else 0")
    screen_id: str = Field(default="Screen_1_IMAX", description="Screen identifier")
    screen_tier: str = Field(default="IMAX", description="Screen tier: IMAX, Premium, Standard, VIP_Gold")
    screen_capacity: int = Field(default=420, ge=50, le=600, description="Screen seat capacity")
    ticket_price: float = Field(default=22.50, ge=5.0, le=80.0, description="Ticket price in USD")
    time_slot: str = Field(default="Night_Prime", description="Morning, Matinee, Evening_Prime, Night_Prime, Late_Night")
    show_hour: int = Field(default=20, ge=8, le=23, description="Hour of show (24h clock)")
    day_of_week: int = Field(default=5, ge=0, le=6, description="0=Monday, 6=Sunday")
    is_weekend: int = Field(default=1, ge=0, le=1, description="1 for Friday/Saturday/Sunday")
    is_holiday: int = Field(default=0, ge=0, le=1, description="1 if public holiday")
    competing_releases_count: int = Field(default=4, ge=1, le=15, description="Concurrent active movies")
    prebooking_pace_t24: int = Field(default=220, ge=0, description="Tickets already booked 24h prior to show")
    historical_movie_avg_occupancy: float = Field(default=0.65, ge=0.0, le=1.0, description="Past occupancy for this movie")


class ShowPredictionResponse(BaseModel):
    predicted_tickets: int
    screen_capacity: int
    predicted_occupancy_pct: float
    confidence_interval_80: Dict[str, Any]
    is_cold_start: bool
    cold_start_warning: Optional[str] = None


@app.get("/", response_class=FileResponse)
def serve_dashboard():
    """Serve the interactive web decision support interface."""
    index_path = STATIC_DIR / "index.html"
    return FileResponse(index_path)


@app.get("/health")
def health_check():
    """Health status and loaded model artifact metadata."""
    try:
        meta = get_metadata()
        return {
            "status": "healthy",
            "model_name": meta.get("model_name"),
            "algorithm": meta.get("algorithm"),
            "forecast_horizon": meta.get("forecast_horizon"),
            "training_samples": meta.get("training_samples"),
            "test_samples": meta.get("test_samples"),
            "test_date_range": meta.get("test_date_range"),
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/metrics")
def get_evaluation_metrics():
    """Return benchmarked evaluation metrics on the held-out test set."""
    meta = get_metadata()
    return {
        "champion_model": meta.get("model_name"),
        "model_comparison": meta.get("model_comparison"),
        "test_samples": meta.get("test_samples"),
        "test_date_range": meta.get("test_date_range"),
    }


@app.post("/predict", response_model=ShowPredictionResponse)
def predict_show_demand(payload: ShowPredictionInput):
    """Predict show tickets sold bounded by capacity with 80% confidence interval."""
    model = get_model()
    data_dict = payload.model_dump()
    df_row = pd.DataFrame([data_dict])[RAW_FEATURE_COLS]
    
    preds, lower, upper = model.predict_with_intervals(df_row, confidence=0.80)
    pred_tickets = int(preds[0])
    lower_b = int(lower[0])
    upper_b = int(upper[0])
    capacity = int(payload.screen_capacity)
    occ_pct = round((pred_tickets / capacity) * 100, 1)
    
    is_cold = bool(payload.is_cold_start or payload.days_since_release <= 3)
    warning = None
    if is_cold:
        warning = f"Cold-start: Theatrical run is day {payload.days_since_release} (<= 3 days). Expect wider demand variance."

    return ShowPredictionResponse(
        predicted_tickets=pred_tickets,
        screen_capacity=capacity,
        predicted_occupancy_pct=occ_pct,
        confidence_interval_80={
            "lower_bound": lower_b,
            "upper_bound": upper_b,
            "interval_width_tickets": upper_b - lower_b,
        },
        is_cold_start=is_cold,
        cold_start_warning=warning,
    )


@app.post("/explain")
def explain_prediction(payload: ShowPredictionInput):
    """Full decision-support breakdown: prediction, confidence bounds, key drivers, staffing, and screen swap advisory."""
    model = get_model()
    data_dict = payload.model_dump()
    df_row = pd.DataFrame([data_dict])[RAW_FEATURE_COLS]
    
    preds, lower, upper = model.predict_with_intervals(df_row, confidence=0.80)
    explanation = explain_single_prediction(
        show_data=data_dict,
        predicted_tickets=int(preds[0]),
        lower_bound=int(lower[0]),
        upper_bound=int(upper[0]),
    )
    return explanation


@app.post("/predict/batch")
def predict_batch(shows: List[ShowPredictionInput]):
    """Batch prediction for multi-screen schedule optimization."""
    model = get_model()
    rows = [s.model_dump() for s in shows]
    df_batch = pd.DataFrame(rows)[RAW_FEATURE_COLS]
    preds, lower, upper = model.predict_with_intervals(df_batch, confidence=0.80)
    
    results = []
    for i, s in enumerate(shows):
        cap = int(s.screen_capacity)
        t_pred = int(preds[i])
        results.append({
            "movie_title": s.movie_title,
            "screen_id": s.screen_id,
            "time_slot": s.time_slot,
            "screen_capacity": cap,
            "predicted_tickets": t_pred,
            "predicted_occupancy_pct": round((t_pred / cap) * 100, 1),
            "lower_80": int(lower[i]),
            "upper_80": int(upper[i]),
            "is_cold_start": bool(s.is_cold_start or s.days_since_release <= 3),
        })
    return {"total_shows": len(results), "predictions": results}
