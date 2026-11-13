"""
Synthetic Data Generator for Cinema Ticket Demand Forecasting.
Generates realistic show-level booking records reflecting box-office lifecycle decay,
weekend/holiday spikes, screen capacity saturation, and T-24h prebooking pace.
"""
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from src.config import (
    DATA_PATH,
    DATASET_METADATA_PATH,
    GENRES,
    RANDOM_SEED,
    SCREEN_CONFIGS,
    SHOWTIME_SLOTS,
)

FICTIONAL_MOVIES = [
    # Action & Sci-Fi
    {"title": "Chronicles of Nebula", "genre": "Action_SciFi", "rating": 8.4, "star_power": 9.0, "runtime": 148, "release_date": "2026-01-09"},
    {"title": "Cybernetic Horizon", "genre": "Action_SciFi", "rating": 7.6, "star_power": 7.5, "runtime": 132, "release_date": "2026-02-13"},
    {"title": "Velocity Rush", "genre": "Action_SciFi", "rating": 6.8, "star_power": 8.0, "runtime": 115, "release_date": "2026-03-27"},
    {"title": "Aegis Protocol", "genre": "Action_SciFi", "rating": 8.7, "star_power": 9.5, "runtime": 155, "release_date": "2026-05-01"},
    {"title": "Solar Vanguard", "genre": "Action_SciFi", "rating": 7.2, "star_power": 6.8, "runtime": 124, "release_date": "2026-06-19"},
    {"title": "Quantum Rift", "genre": "Action_SciFi", "rating": 8.1, "star_power": 8.8, "runtime": 140, "release_date": "2026-07-24"},
    
    # Drama
    {"title": "The Silent Canvas", "genre": "Drama", "rating": 8.9, "star_power": 6.5, "runtime": 128, "release_date": "2026-01-16"},
    {"title": "Echoes of Eternity", "genre": "Drama", "rating": 8.2, "star_power": 7.0, "runtime": 135, "release_date": "2026-03-06"},
    {"title": "The Glass Architect", "genre": "Drama", "rating": 7.4, "star_power": 6.0, "runtime": 118, "release_date": "2026-04-17"},
    {"title": "Autumn Letters", "genre": "Drama", "rating": 7.9, "star_power": 5.8, "runtime": 122, "release_date": "2026-06-05"},
    {"title": "Winter in Florence", "genre": "Drama", "rating": 8.5, "star_power": 7.2, "runtime": 142, "release_date": "2026-08-07"},
    
    # Comedy
    {"title": "Office Disaster", "genre": "Comedy", "rating": 6.9, "star_power": 7.1, "runtime": 98, "release_date": "2026-01-23"},
    {"title": "The Accidental Diplomat", "genre": "Comedy", "rating": 7.5, "star_power": 7.8, "runtime": 105, "release_date": "2026-03-13"},
    {"title": "Laugh Protocol", "genre": "Comedy", "rating": 7.1, "star_power": 6.2, "runtime": 94, "release_date": "2026-05-15"},
    {"title": "Roommate Roulette", "genre": "Comedy", "rating": 6.4, "star_power": 5.5, "runtime": 92, "release_date": "2026-07-03"},
    
    # Horror / Thriller
    {"title": "The Whispering Hollow", "genre": "Horror_Thriller", "rating": 7.8, "star_power": 5.9, "runtime": 104, "release_date": "2026-02-06"},
    {"title": "Midnight Corridor", "genre": "Horror_Thriller", "rating": 7.1, "star_power": 6.1, "runtime": 99, "release_date": "2026-04-03"},
    {"title": "Phantom Frequency", "genre": "Horror_Thriller", "rating": 6.7, "star_power": 5.2, "runtime": 101, "release_date": "2026-06-26"},
    {"title": "The 13th Room", "genre": "Horror_Thriller", "rating": 8.0, "star_power": 7.4, "runtime": 112, "release_date": "2026-08-14"},
    
    # Animation & Family
    {"title": "Penguin Kingdom", "genre": "Animation_Family", "rating": 8.3, "star_power": 7.0, "runtime": 95, "release_date": "2026-02-20"},
    {"title": "Cloud Riders", "genre": "Animation_Family", "rating": 8.6, "star_power": 8.2, "runtime": 102, "release_date": "2026-04-24"},
    {"title": "Tiny Dragon Adventures", "genre": "Animation_Family", "rating": 7.7, "star_power": 6.0, "runtime": 88, "release_date": "2026-07-10"},
    
    # Romantic Drama
    {"title": "Parallel Hearts", "genre": "Romantic_Drama", "rating": 7.8, "star_power": 7.9, "runtime": 119, "release_date": "2026-02-14"},
    {"title": "Sunset by the Seine", "genre": "Romantic_Drama", "rating": 7.3, "star_power": 6.4, "runtime": 110, "release_date": "2026-05-22"},
    {"title": "Chasing Stardust", "genre": "Romantic_Drama", "rating": 8.1, "star_power": 7.5, "runtime": 126, "release_date": "2026-07-31"},
]

# Major regional holidays in 2026 simulation
HOLIDAYS_2026 = {
    "2026-01-01": "New Year Day",
    "2026-01-26": "Republic Day",
    "2026-02-14": "Valentine Special",
    "2026-03-04": "Holi Festival",
    "2026-04-03": "Good Friday",
    "2026-04-14": "Spring Holiday",
    "2026-05-01": "Labor Day",
    "2026-08-15": "Independence Day",
}


def generate_dataset(start_date="2026-01-01", end_date="2026-08-31", seed=RANDOM_SEED) -> pd.DataFrame:
    """Generate show-level cinema ticketing data chronologically."""
    np.random.seed(seed)
    
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days + 1
    
    records = []
    show_id_counter = 10001

    # Keep track of movie historical show bookings to build dynamic lag features without leakage
    movie_show_history = {m["title"]: [] for m in FICTIONAL_MOVIES}
    
    current_day = start_dt
    while current_day <= end_dt:
        date_str = current_day.strftime("%Y-%m-%d")
        day_of_week = current_day.weekday()  # 0=Monday, 6=Sunday
        is_weekend = 1 if day_of_week in [4, 5, 6] else 0  # Fri, Sat, Sun
        is_holiday = 1 if date_str in HOLIDAYS_2026 else 0
        
        # Determine currently playing movies on this day
        active_movies = []
        for movie in FICTIONAL_MOVIES:
            rel_dt = datetime.strptime(movie["release_date"], "%Y-%m-%d")
            days_active = (current_day - rel_dt).days
            # Movies play for up to 45 days after release
            if 0 <= days_active <= 45:
                active_movies.append({**movie, "days_active": days_active})
        
        if not active_movies:
            # Fallback if window starts before first release
            active_movies = [{**FICTIONAL_MOVIES[0], "days_active": 1}]
            
        competing_count = len(active_movies)

        # Allocate movies to screens based on movie age, star power, and screen capacity
        # Newer, high-star movies get larger screens (Screen 1 IMAX, Screen 2 Large)
        sorted_active = sorted(active_movies, key=lambda x: (x["days_active"] <= 7, x["star_power"], x["rating"]), reverse=True)

        for screen_name, screen_info in SCREEN_CONFIGS.items():
            capacity = screen_info["capacity"]
            screen_tier = screen_info["tier"]
            base_multiplier = screen_info["base_price_multiplier"]
            
            # Select movie for this screen today
            if screen_tier == "IMAX":
                # IMAX prioritizes high-budget Action/Sci-Fi or top blockbusters
                imax_candidates = [m for m in sorted_active if m["genre"] in ["Action_SciFi", "Animation_Family"]]
                assigned_movie = imax_candidates[0] if imax_candidates else sorted_active[0]
            elif screen_tier == "VIP_Gold":
                # VIP prioritizes top-rated dramas or romantic movies
                vip_candidates = [m for m in sorted_active if m["genre"] in ["Drama", "Romantic_Drama", "Action_SciFi"]]
                assigned_movie = vip_candidates[0] if vip_candidates else sorted_active[min(1, len(sorted_active)-1)]
            else:
                assigned_idx = hash(screen_name + date_str) % len(sorted_active)
                assigned_movie = sorted_active[assigned_idx]
            
            days_since_rel = assigned_movie["days_active"]
            is_cold_start = 1 if days_since_rel <= 3 else 0

            # Schedule shows across time slots
            for slot in SHOWTIME_SLOTS:
                slot_id = slot["slot_id"]
                show_hour = slot["hour"]
                time_str = slot["time"]
                slot_weight = slot["demand_weight"]
                
                show_datetime_str = f"{date_str} {time_str}"
                
                # Ticket price calculation with weekend & format dynamics
                base_price = 12.50
                price = round(base_price * base_multiplier * (1.15 if is_weekend else 1.0) * (1.10 if is_holiday else 1.0), 2)
                
                # Demand calculation physics:
                # 1. Base demand factor by genre and rating
                genre_factor = {
                    "Action_SciFi": 0.72,
                    "Animation_Family": 0.68 if (is_weekend or slot_id in ["Morning", "Matinee"]) else 0.45,
                    "Drama": 0.55,
                    "Comedy": 0.58,
                    "Horror_Thriller": 0.62 if slot_id in ["Night_Prime", "Late_Night"] else 0.40,
                    "Romantic_Drama": 0.54 if (is_weekend or is_holiday) else 0.42,
                }.get(assigned_movie["genre"], 0.50)
                
                # 2. Movie lifecycle decay: highest on opening weekend (days 0-3), decaying smoothly
                decay = np.exp(-0.045 * max(0, days_since_rel - 3))
                if is_cold_start:
                    decay = 1.08  # Opening spike
                    
                # 3. Quality & star power bonus
                quality_mult = (assigned_movie["rating"] / 7.5) * 0.45 + (assigned_movie["star_power"] / 7.5) * 0.55
                
                # 4. Day and slot multiplier
                day_mult = 1.35 if day_of_week in [4, 5] else (1.20 if day_of_week == 6 else (0.90 if day_of_week == 1 else 0.75))
                if is_holiday:
                    day_mult *= 1.40
                
                # Composite latent demand proportion
                latent_occupancy = genre_factor * decay * quality_mult * (slot_weight / 1.1) * (day_mult / 1.0)
                
                # Competing titles cannibalization factor
                competition_penalty = 1.0 - (0.015 * min(competing_count, 10))
                latent_occupancy *= competition_penalty
                
                # Add realistic noise (log-normal variation)
                noise = np.random.normal(0, 0.08)
                expected_occupancy = float(np.clip(latent_occupancy + noise, 0.05, 0.98))
                
                # Pre-booking pace at T-24h (prior advance ticket bookings)
                # Typically 45% to 75% of final tickets are booked 24h prior to show
                prebook_ratio = np.random.uniform(0.48, 0.72)
                final_tickets = int(round(expected_occupancy * capacity))
                final_tickets = min(capacity, max(3, final_tickets))
                
                prebooking_pace_t24 = int(round(final_tickets * prebook_ratio * np.random.uniform(0.92, 1.08)))
                prebooking_pace_t24 = min(prebooking_pace_t24, final_tickets)
                
                # Calculate historical movie rolling average up to prior day (leakage-free)
                history = movie_show_history[assigned_movie["title"]]
                if len(history) > 0:
                    hist_movie_avg_occ = float(np.mean(history[-20:]))  # Last 20 shows average
                else:
                    hist_movie_avg_occ = float(genre_factor * 0.9)  # Cold start prior
                    
                # Actual final occupancy rate
                final_occupancy_rate = round(final_tickets / capacity, 4)

                records.append({
                    "show_id": show_id_counter,
                    "show_datetime": show_datetime_str,
                    "date": date_str,
                    "movie_title": assigned_movie["title"],
                    "genre": assigned_movie["genre"],
                    "critic_rating": assigned_movie["rating"],
                    "star_power": assigned_movie["star_power"],
                    "runtime_minutes": assigned_movie["runtime"],
                    "days_since_release": days_since_rel,
                    "is_cold_start": is_cold_start,
                    "screen_id": screen_name,
                    "screen_tier": screen_tier,
                    "screen_capacity": capacity,
                    "ticket_price": price,
                    "time_slot": slot_id,
                    "show_hour": show_hour,
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                    "is_holiday": is_holiday,
                    "competing_releases_count": competing_count,
                    "prebooking_pace_t24": prebooking_pace_t24,
                    "historical_movie_avg_occupancy": round(hist_movie_avg_occ, 4),
                    "tickets_sold": final_tickets,
                    "occupancy_rate": final_occupancy_rate,
                })
                
                # Update movie history for future days
                movie_show_history[assigned_movie["title"]].append(final_occupancy_rate)
                show_id_counter += 1
                
        current_day += timedelta(days=1)
        
    df = pd.DataFrame(records)
    # Sort strictly chronologically
    df["show_datetime"] = pd.to_datetime(df["show_datetime"])
    df = df.sort_values(by="show_datetime").reset_index(drop=True)
    return df


def save_dataset_and_metadata(df: pd.DataFrame):
    """Save dataset to CSV and generate metadata schema summary."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    
    metadata = {
        "dataset_name": "Cinema Hall Ticket Demand Show-Level Dataset",
        "description": "Chronological show-level ticketing observations for multi-screen cinema demand forecasting.",
        "record_count": len(df),
        "date_range_start": str(df["date"].min()),
        "date_range_end": str(df["date"].max()),
        "target_variables": ["tickets_sold", "occupancy_rate"],
        "forecast_horizon": "T-24h (24 hours prior to showtime)",
        "prediction_unit": "Individual Show screening (Screen, Showtime, Movie, Date)",
        "num_screens": len(SCREEN_CONFIGS),
        "total_movies": df["movie_title"].nunique(),
        "genres": df["genre"].unique().tolist(),
        "cold_start_shows": int((df["is_cold_start"] == 1).sum()),
        "mature_shows": int((df["is_cold_start"] == 0).sum()),
        "features": {
            "temporal": ["show_hour", "day_of_week", "is_weekend", "is_holiday", "time_slot"],
            "movie_attributes": ["genre", "critic_rating", "star_power", "runtime_minutes", "days_since_release", "is_cold_start"],
            "screen_attributes": ["screen_id", "screen_tier", "screen_capacity", "ticket_price"],
            "signals_at_t24": ["prebooking_pace_t24", "historical_movie_avg_occupancy", "competing_releases_count"],
        },
        "license": "Creative Commons Zero (CC0) / Public Domain dedication for assessment use. No PII or proprietary cinema data.",
        "generated_at": datetime.now().isoformat(),
    }
    
    with open(DATASET_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset generated successfully: {len(df)} records saved to {DATA_PATH}")
    print(f"Metadata saved to {DATASET_METADATA_PATH}")


if __name__ == "__main__":
    df_generated = generate_dataset()
    save_dataset_and_metadata(df_generated)
