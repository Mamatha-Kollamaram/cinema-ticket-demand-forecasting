"""
Comprehensive Evaluation, Slicing, and Error Analysis for Cinema Demand Forecasting.
Generates evaluation tables, cold-start diagnostics, and metric charts.
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import (
    DATA_PATH,
    MODEL_PATH,
    METADATA_PATH,
    ARTIFACTS_DIR,
)
from src.features import (
    RAW_FEATURE_COLS,
    TARGET_COL,
    chronological_train_val_test_split,
)
from src.models import compute_forecasting_metrics


def run_full_evaluation():
    print("Executing comprehensive evaluation and error analysis...")
    df = pd.read_csv(DATA_PATH)
    _, _, test_df = chronological_train_val_test_split(df, 0.70, 0.15)
    
    # Load champion model
    champion_model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)
        
    X_test = test_df[RAW_FEATURE_COLS]
    y_test = test_df[TARGET_COL].values
    caps = test_df["screen_capacity"].values
    
    preds, lower, upper = champion_model.predict_with_intervals(X_test, confidence=0.80)
    
    # Add predictions to test dataframe for segmented slicing
    test_eval = test_df.copy()
    test_eval["predicted_tickets"] = preds
    test_eval["pred_lower_80"] = lower
    test_eval["pred_upper_80"] = upper
    test_eval["abs_error"] = np.abs(test_eval["tickets_sold"] - test_eval["predicted_tickets"])
    test_eval["sq_error"] = (test_eval["tickets_sold"] - test_eval["predicted_tickets"]) ** 2
    test_eval["predicted_occupancy"] = test_eval["predicted_tickets"] / test_eval["screen_capacity"]
    
    overall_metrics = compute_forecasting_metrics(y_test, preds, caps)
    
    print("\n========== OVERALL TEST SET METRICS ==========")
    for k, v in overall_metrics.items():
        print(f"  {k}: {v}")
    
    # 1. Cold-start vs Mature shows slice
    cold_slice = test_eval.groupby("is_cold_start").agg(
        shows=("show_id", "count"),
        avg_actual_tickets=("tickets_sold", "mean"),
        avg_predicted_tickets=("predicted_tickets", "mean"),
        mae=("abs_error", "mean"),
        rmse=("sq_error", lambda x: np.sqrt(np.mean(x))),
        wape_pct=("abs_error", lambda x: np.sum(x) / np.sum(test_eval.loc[x.index, "tickets_sold"]) * 100),
    ).reset_index()
    cold_slice["status"] = cold_slice["is_cold_start"].map({1: "Cold-Start (Days 0-3)", 0: "Mature Release (>3 Days)"})
    
    # 2. Screen Tier slice
    tier_slice = test_eval.groupby("screen_tier").agg(
        shows=("show_id", "count"),
        capacity=("screen_capacity", "first"),
        avg_actual=("tickets_sold", "mean"),
        mae=("abs_error", "mean"),
        occupancy_mae_pct=("abs_error", lambda x: np.mean(x / test_eval.loc[x.index, "screen_capacity"]) * 100),
    ).reset_index()
    
    # 3. Weekend vs Weekday slice
    day_slice = test_eval.groupby("is_weekend").agg(
        shows=("show_id", "count"),
        avg_actual=("tickets_sold", "mean"),
        mae=("abs_error", "mean"),
        wape_pct=("abs_error", lambda x: np.sum(x) / np.sum(test_eval.loc[x.index, "tickets_sold"]) * 100),
    ).reset_index()
    day_slice["day_type"] = day_slice["is_weekend"].map({1: "Weekend / Holiday Peak", 0: "Weekday Regular"})
    
    print("\n========== SUBGROUP: COLD-START VS MATURE RELEASES ==========")
    print(cold_slice[["status", "shows", "mae", "rmse", "wape_pct"]].to_string(index=False))
    
    print("\n========== SUBGROUP: SCREEN TIER ACCURACY ==========")
    print(tier_slice.to_string(index=False))
    
    print("\n========== SUBGROUP: DAY TYPE (WEEKEND VS WEEKDAY) ==========")
    print(day_slice[["day_type", "shows", "avg_actual", "mae", "wape_pct"]].to_string(index=False))
    
    # Generate Visualizations
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Figure 1: Actual vs Predicted Scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, preds, alpha=0.35, color="#1f77b4", edgecolors="none", s=25, label="Test Shows")
    max_val = max(y_test.max(), preds.max()) + 10
    ax.plot([0, max_val], [0, max_val], color="#e74c3c", linestyle="--", linewidth=1.8, label="Ideal 45° Line")
    ax.set_xlabel("Actual Tickets Sold", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Tickets Sold", fontsize=11, fontweight="bold")
    ax.set_title("Actual vs. Predicted Show Tickets (Held-Out Test Set)", fontsize=12, fontweight="bold")
    ax.legend(frameon=True)
    plt.tight_layout()
    scatter_path = ARTIFACTS_DIR / "actual_vs_predicted.png"
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    print(f"\nSaved plot: {scatter_path}")
    
    # Figure 2: Model Comparison Bar Chart
    comp_data = meta.get("model_comparison", {})
    if comp_data:
        models = list(comp_data.keys())
        maes = [comp_data[m]["test_metrics"]["MAE"] for m in models]
        wapes = [comp_data[m]["test_metrics"]["WAPE_pct"] for m in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax1.bar(models, maes, color=["#95a5a6", "#3498db", "#2ecc71"], width=0.55)
        ax1.set_ylabel("Mean Absolute Error (Tickets)", fontweight="bold")
        ax1.set_title("Test MAE by Model Architecture", fontweight="bold")
        for i, v in enumerate(maes):
            ax1.text(i, v + 0.8, f"{v}", ha="center", fontweight="bold")
            
        ax2.bar(models, wapes, color=["#95a5a6", "#3498db", "#2ecc71"], width=0.55)
        ax2.set_ylabel("WAPE (%)", fontweight="bold")
        ax2.set_title("Test WAPE (%) by Model Architecture", fontweight="bold")
        for i, v in enumerate(wapes):
            ax2.text(i, v + 1.0, f"{v}%", ha="center", fontweight="bold")
            
        plt.tight_layout()
        comp_path = ARTIFACTS_DIR / "model_comparison_metrics.png"
        plt.savefig(comp_path, dpi=200)
        plt.close()
        print(f"Saved plot: {comp_path}")
        
    return {
        "overall": overall_metrics,
        "cold_start": cold_slice.to_dict(orient="records"),
        "tier": tier_slice.to_dict(orient="records"),
        "day_type": day_slice.to_dict(orient="records"),
    }


if __name__ == "__main__":
    run_full_evaluation()

