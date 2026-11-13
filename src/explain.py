"""
Explainability, Decision Support, and Feature Importance Module.
Provides local prediction explanations, key feature drivers, and operational advisory.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from src.config import STAFF_ALLOCATION_RULES, SCREEN_CONFIGS
from src.features import RAW_FEATURE_COLS


def get_feature_importances(model, X_val: pd.DataFrame, y_val: pd.Series) -> List[Dict[str, Any]]:
    """Compute permutation feature importances for champion model."""
    result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )
    
    importances = []
    for col, mean_imp in zip(RAW_FEATURE_COLS, result.importances_mean):
        importances.append({
            "feature": col,
            "importance_score": round(float(mean_imp), 4),
        })
        
    # Sort descending
    importances.sort(key=lambda x: x["importance_score"], reverse=True)
    return importances


def generate_staff_recommendation(occupancy_rate: float, predicted_tickets: int) -> Dict[str, Any]:
    """Provide rule-based operational staffing recommendations based on occupancy tier."""
    if occupancy_rate <= STAFF_ALLOCATION_RULES["low_demand"]["max_occupancy"]:
        tier = "Low Demand (<= 35% Occupancy)"
        staff = STAFF_ALLOCATION_RULES["low_demand"]
    elif occupancy_rate <= STAFF_ALLOCATION_RULES["medium_demand"]["max_occupancy"]:
        tier = "Moderate Demand (36% - 70% Occupancy)"
        staff = STAFF_ALLOCATION_RULES["medium_demand"]
    else:
        tier = "Peak / High Demand (> 70% Occupancy)"
        staff = STAFF_ALLOCATION_RULES["high_demand"]
        
    return {
        "demand_tier": tier,
        "concession_staff": staff["concession_staff"],
        "ushers": staff["ushers"],
        "cleaning_crew": staff["cleaning_staff"],
        "total_staff_required": staff["concession_staff"] + staff["ushers"] + staff["cleaning_staff"],
    }


def generate_screen_allocation_advisory(
    screen_capacity: int,
    predicted_tickets: int,
    occupancy_rate: float,
    current_screen: str,
) -> Dict[str, Any]:
    """Analyze whether the show should be swapped to a different screen."""
    if occupancy_rate > 0.88:
        # Check if larger screen exists
        larger_screens = [
            (name, cfg["capacity"])
            for name, cfg in SCREEN_CONFIGS.items()
            if cfg["capacity"] > screen_capacity
        ]
        if larger_screens:
            best_target = max(larger_screens, key=lambda x: x[1])
            return {
                "recommendation": "SWAP_UPGRADE",
                "risk": "Turnaway / Capacity Saturation Risk",
                "message": (
                    f"Show is projected at {occupancy_rate*100:.1f}% capacity ({predicted_tickets}/{screen_capacity} seats). "
                    f"High probability of selling out early. Consider re-allocating to {best_target[0]} ({best_target[1]} seats) "
                    "to capture unmet demand."
                ),
            }
        else:
            return {
                "recommendation": "MAX_CAPACITY",
                "risk": "Near Sellout",
                "message": f"Show is projected at {occupancy_rate*100:.1f}% capacity on the venue's largest screen. Ensure express concession lanes.",
            }
    elif occupancy_rate < 0.25 and screen_capacity >= 250:
        smaller_screens = [
            (name, cfg["capacity"])
            for name, cfg in SCREEN_CONFIGS.items()
            if cfg["capacity"] < screen_capacity and cfg["capacity"] >= predicted_tickets * 1.3
        ]
        if smaller_screens:
            best_target = min(smaller_screens, key=lambda x: abs(x[1] - predicted_tickets * 1.5))
            return {
                "recommendation": "DOWNSIZE_SCREEN",
                "risk": "Low Hall Utilization",
                "message": (
                    f"Show projected at only {occupancy_rate*100:.1f}% occupancy ({predicted_tickets}/{screen_capacity} seats) on large screen. "
                    f"Consider swapping to {best_target[0]} ({best_target[1]} seats) to conserve HVAC/projection energy."
                ),
            }
            
    return {
        "recommendation": "OPTIMAL_SCREEN",
        "risk": "Balanced",
        "message": f"Current screen allocation is optimal ({occupancy_rate*100:.1f}% occupancy).",
    }


def explain_single_prediction(
    show_data: Dict[str, Any],
    predicted_tickets: int,
    lower_bound: int,
    upper_bound: int,
) -> Dict[str, Any]:
    """Generate interpretable explanation, drivers, and cold-start diagnostic for one show."""
    capacity = int(show_data["screen_capacity"])
    occupancy = round(predicted_tickets / capacity, 4)
    
    # Assess key drivers
    drivers = []
    
    # Driver 1: Pre-booking velocity
    prebook = show_data.get("prebooking_pace_t24", 0)
    prebook_ratio = prebook / capacity
    if prebook_ratio > 0.40:
        drivers.append({
            "factor": "Strong Advance Pre-Booking",
            "impact": "Positive (++): Rapid ticket sales 24h prior to show indicates strong organic demand.",
            "value": f"{prebook} tickets already reserved ({prebook_ratio*100:.1f}% of hall)",
        })
    else:
        drivers.append({
            "factor": "Muted Pre-Booking Velocity",
            "impact": "Neutral / Downward (-): Walk-in demand will need to make up the difference.",
            "value": f"{prebook} tickets reserved ({prebook_ratio*100:.1f}% of hall)",
        })
        
    # Driver 2: Timing & Calendar
    if show_data.get("is_weekend", 0) == 1:
        drivers.append({
            "factor": "Weekend Prime Timing",
            "impact": "Positive (++): Weekend leisure crowds drive higher attendance across all genres.",
            "value": f"Weekend (Slot: {show_data.get('time_slot', 'N/A')})",
        })
    else:
        drivers.append({
            "factor": "Weekday Timing",
            "impact": "Discount / Routine: Weekday showings exhibit lower casual footfall.",
            "value": f"Weekday (Slot: {show_data.get('time_slot', 'N/A')})",
        })
        
    # Driver 3: Movie Lifecycle & Cold Start
    days_rel = show_data.get("days_since_release", 0)
    is_cold = show_data.get("is_cold_start", 0) == 1
    
    cold_start_warning = None
    if is_cold:
        cold_start_warning = {
            "is_cold_start": True,
            "status": "COLD_START_ACTIVE",
            "message": (
                f"Movie has only been in theaters for {days_rel} day(s). Historical same-movie empirical show data is minimal. "
                "Forecast leans on genre priors, star power, and opening weekend multipliers. "
                "Recommendation: Treat prediction intervals as wider; maintain on-call usher backup."
            ),
        }
        drivers.append({
            "factor": "Opening Weekend Surge (Cold-Start)",
            "impact": "Elevated Initial Demand: High opening momentum, offset by higher variance.",
            "value": f"Day {days_rel} since theatrical release",
        })
    else:
        drivers.append({
            "factor": "Mature Run Lifecycle",
            "impact": "Stable / Predictable Decay: Grounded in robust historical show occupancy.",
            "value": f"Day {days_rel} of theatrical run",
        })
        
    # Staffing recommendation
    staffing = generate_staff_recommendation(occupancy, predicted_tickets)
    
    # Screen allocation advisory
    screen_advisory = generate_screen_allocation_advisory(
        capacity, predicted_tickets, occupancy, show_data.get("screen_id", "Screen")
    )
    
    return {
        "predicted_tickets": predicted_tickets,
        "screen_capacity": capacity,
        "predicted_occupancy_pct": round(occupancy * 100, 1),
        "confidence_interval_80": {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "interval_width_tickets": upper_bound - lower_bound,
        },
        "cold_start_diagnostic": cold_start_warning,
        "key_drivers": drivers,
        "staffing_recommendation": staffing,
        "screen_allocation_advisory": screen_advisory,
        "governance_note": (
            "Decision-Support System: Human cinema floor managers retain final discretion. "
            "Predictions are updated at T-24h and should be re-evaluated alongside live weather or localized events."
        ),
    }
