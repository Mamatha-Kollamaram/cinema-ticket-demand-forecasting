"""
Configuration and constants for Cinema Ticket Demand Forecasting.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
STATIC_DIR = BASE_DIR / "static"

DATA_PATH = DATA_DIR / "cinema_shows_dataset.csv"
DATASET_METADATA_PATH = DATA_DIR / "dataset_metadata.json"
MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
FEATURE_NAMES_PATH = ARTIFACTS_DIR / "feature_names.json"

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Screen Configurations
SCREEN_CONFIGS = {
    "Screen_1_IMAX": {"capacity": 420, "tier": "IMAX", "base_price_multiplier": 1.45},
    "Screen_2_Large": {"capacity": 300, "tier": "Premium", "base_price_multiplier": 1.20},
    "Screen_3_MidA": {"capacity": 220, "tier": "Standard", "base_price_multiplier": 1.00},
    "Screen_4_MidB": {"capacity": 180, "tier": "Standard", "base_price_multiplier": 1.00},
    "Screen_5_VIP": {"capacity": 80, "tier": "VIP_Gold", "base_price_multiplier": 2.10},
    "Screen_6_Classic": {"capacity": 150, "tier": "Standard", "base_price_multiplier": 0.90},
}

# Showtime Slots (24h clock)
SHOWTIME_SLOTS = [
    {"slot_id": "Morning", "time": "10:00", "hour": 10, "demand_weight": 0.65},
    {"slot_id": "Matinee", "time": "13:30", "hour": 13, "demand_weight": 0.85},
    {"slot_id": "Evening_Prime", "time": "17:30", "hour": 17, "demand_weight": 1.40},
    {"slot_id": "Night_Prime", "time": "20:45", "hour": 20, "demand_weight": 1.50},
    {"slot_id": "Late_Night", "time": "23:30", "hour": 23, "demand_weight": 0.90},
]

# Movie Genres and Demand Baseline
GENRES = [
    "Action_SciFi",
    "Drama",
    "Comedy",
    "Horror_Thriller",
    "Animation_Family",
    "Romantic_Drama",
]

# Forecast Horizon Definition
# Predictions are generated 24 hours prior to scheduled showtime (T-24h).
# Features reflect only data known before or at T-24h.
FORECAST_HORIZON_HOURS = 24

# Staffing Rules (Decision Support Allocation)
STAFF_ALLOCATION_RULES = {
    "low_demand": {"max_occupancy": 0.35, "concession_staff": 2, "ushers": 1, "cleaning_staff": 1},
    "medium_demand": {"max_occupancy": 0.70, "concession_staff": 4, "ushers": 2, "cleaning_staff": 2},
    "high_demand": {"max_occupancy": 1.00, "concession_staff": 7, "ushers": 4, "cleaning_staff": 3},
}

