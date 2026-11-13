# Technical Report: Cinema Ticket Demand Forecasting

**Assessment:** Advanced Individual Project Assessment — AI & Machine Learning  
**Organization:** ALGIONS AI LABS PRIVATE LIMITED  
**Domain:** Cinema Hall Ticket Booking  
**Author:** AI & ML Engineering Candidate  
**Date:** November 2026  

---

## Executive Summary

This report presents a reproducible, capacity-aware Machine Learning decision-support system for **Cinema Ticket Demand Forecasting**. In cinema operations, ticket bookings fluctuate dynamically based on movie lifecycle decay, weekday-versus-weekend leisure patterns, showtime convenience, pricing tiers, and advance pre-booking momentum. Cinema floor managers must finalize screen allocations and staffing rosters (concessions, floor ushers, cleaning teams) 24 hours prior to showtime ($T-24\text{h}$).

To address this challenge, we developed:
1. A **chronological, leakage-free modeling pipeline** operating on show-level screening events.
2. A **Capacity-Bounded Gradient Boosted Regressor** that outperforms a naive capacity-adjusted historical baseline by **82.3%** in Mean Absolute Error (MAE reduced from $36.37$ to $6.44$ tickets) and achieves a Weighted Absolute Percentage Error (WAPE) of **$9.33\%$** on held-out test data.
3. Strict enforcement of **physical capacity truncation** ($0 \le \hat{y} \le \text{screen\_capacity}$), ensuring $0\%$ capacity overrun risk.
4. A **Cold-Start diagnostic module** with widened confidence bounds for new theatrical releases ($\le 3$ days active).
5. A working **FastAPI inference service and interactive Web Dashboard** serving real model predictions, confidence intervals, staffing recommendations, and screen re-allocation advisories.

---

## 1. Problem Framing & Operational Scope

### 1.1 Business Decision & Prediction Unit
* **Target Decision:** Optimize cinema screen allocation (preventing sellout turnaway on small screens while avoiding HVAC energy waste on under-utilized large screens) and schedule floor staff rosters.
* **Prediction Unit:** An individual show screening event:
  $$\text{Show} = (Screen,\, Showtime,\, Movie,\, Date)$$
* **Target Variable:** 
  * Primary Target: $y = \text{tickets\_sold} \in [0,\, \text{screen\_capacity}]$.
  * Derived Target: $\text{occupancy\_rate} = \frac{\text{tickets\_sold}}{\text{screen\_capacity}} \in [0.0,\, 1.0]$.
* **Point-in-Time ($T-24\text{h}$ Forecast Horizon):**
  Predictions are generated 24 hours prior to scheduled showtime. All features are strictly constrained to signals available at or before $T-24\text{h}$ (e.g. advance bookings up to $T-24\text{h}$, historical movie performance on prior days, calendar variables, and screen metadata). No same-day walk-in or post-show features are included.

---

## 2. Data Sourcing & Exploratory Data Analysis (EDA)

### 2.1 Dataset Composition & Synthetic Generation
To satisfy assessment licensing and data privacy requirements while reflecting realistic cinema operational dynamics, we constructed a synthetic show-level dataset spanning 8 months (January 1, 2026 to August 31, 2026) comprising **7,290 show screening records** across 6 distinct screen halls.

| Screen Identifier | Hall Tier | Physical Capacity | Base Price Multiplier | Operational Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **Screen 1** | IMAX Premium | 420 seats | $1.45\times$ | High-budget Action/Sci-Fi & Blockbusters |
| **Screen 2** | Large Auditorium | 300 seats | $1.20\times$ | Popular releases & family weekend films |
| **Screen 3** | Mid Auditorium A | 220 seats | $1.00\times$ | Standard releases & comedy screenings |
| **Screen 4** | Mid Auditorium B | 180 seats | $1.00\times$ | Indie dramas & second-week releases |
| **Screen 5** | VIP Gold Lounge | 80 seats | $2.10\times$ | Luxury dine-in screenings & romantic dramas |
| **Screen 6** | Classic Screen | 150 seats | $0.90\times$ | Matinee specials & extended runs |

### 2.2 Data Quality & Distribution Checks
* **Missing Values:** Zero null or missing values across all 7,290 records.
* **Physical Bound Verification:** $\min(\text{tickets\_sold}) = 3$, $\max(\text{tickets\_sold}) \le \text{screen\_capacity}$. Across all records, $0$ shows violated capacity.
* **Cold-Start Prevalence:** $80$ shows in the test set ($7.3\%$) fell into the opening 3-day window of a theatrical release.
* **Seasonality & Day Type:** Shows on Friday evening, Saturday, and Sunday exhibited an average occupancy rate of $61.2\%$ compared to $38.4\%$ for Monday–Thursday shows.

---

## 3. Leakage Prevention & Validation Strategy

### 3.1 Strict Chronological Splitting
Random $k$-fold cross-validation is invalid for forecasting time-series bookings because shuffling allows future box-office trends to leak into past predictions. We enforce a **strict chronological partition**:

```
[------------- TRAIN (70%) -------------] [--- VAL (15%) ---] [--- TEST (15%) ---]
2026-01-01                   2026-06-20  2026-07-26       2026-08-31
(5,103 shows)                            (1,093 shows)     (1,094 shows)
```

* **Training Window:** 2026-01-01 to 2026-06-20 (5,103 shows)
* **Validation Window:** 2026-06-20 to 2026-07-26 (1,093 shows)
* **Held-Out Test Window:** 2026-07-26 to 2026-08-31 (1,094 shows)

Temporal boundaries are strictly disjoint:
$$\max(\text{Train DateTime}) \le \min(\text{Val DateTime}) \le \max(\text{Val DateTime}) \le \min(\text{Test DateTime})$$

---

## 4. Feature Engineering

Features available at $T-24\text{h}$ are grouped into four categories:

1. **Temporal & Calendar Features:** `show_hour`, `day_of_week`, `is_weekend`, `is_holiday`, `time_slot` (Morning, Matinee, Evening, Night, Late Night).
2. **Movie Attributes:** `genre`, `critic_rating`, `star_power`, `runtime_minutes`, `days_since_release`, `is_cold_start`.
3. **Screen Metadata:** `screen_tier`, `screen_capacity`, `ticket_price`.
4. **Pre-Show Signals at $T-24\text{h}$ & Engineered Interactions:**
   * `prebooking_pace_t24`: Confirmed advance bookings 24 hours prior to showtime.
   * `prebook_to_capacity_ratio`: $\frac{\text{prebooking\_pace\_t24}}{\text{screen\_capacity}} \in [0.0, 1.0]$.
   * `decay_factor`: $\exp(-0.045 \times \text{days\_since\_release})$.
   * `historical_movie_avg_occupancy`: Rolling historical occupancy rate of prior shows for the same movie up to $T-24\text{h}$.

Categorical variables are encoded with scikit-learn's `OneHotEncoder(handle_unknown='ignore')`, and continuous numerical features are normalized via `StandardScaler()`.

---

## 5. Model Development & Comparison

We evaluated three architectures under identical chronological train/test conditions:

1. **Naive Capacity-Adjusted Baseline (`CapacityAdjustedBaseline`):**
   Computes empirical mean occupancy grouped by $(\text{genre}, \text{is\_weekend}, \text{time\_slot})$ from training history and multiplies by the show's physical capacity:
   $$\hat{y}_{\text{base}} = \min\left(\text{screen\_capacity},\, \mu_{\text{group}} \times \text{screen\_capacity}\right)$$
2. **Model 1: Capacity-Bounded Ridge Regression (`RidgeRegression`):**
   L2-regularized linear model ($\alpha = 10.0$) with one-hot encoded categoricals and standardized continuous features, with predictions clamped to $[0, \text{screen\_capacity}]$.
3. **Model 2: Capacity-Bounded Gradient Boosting (`GradientBoosting`):**
   `HistGradientBoostingRegressor` with maximum depth, leaf regularization ($31$ max leaves, min $20$ samples per leaf, learning rate $0.07$, $160$ estimators) and physical capacity clipping.

### 5.1 Evaluation Metrics
* **MAE (Mean Absolute Error):** $\frac{1}{N}\sum |y_i - \hat{y}_i|$ (Tickets)
* **RMSE (Root Mean Squared Error):** $\sqrt{\frac{1}{N}\sum (y_i - \hat{y}_i)^2}$ (Tickets)
* **WAPE (Weighted Absolute Percentage Error):** $\frac{\sum |y_i - \hat{y}_i|}{\sum y_i} \times 100\%$
* **Occupancy MAE:** $\frac{1}{N}\sum \frac{|y_i - \hat{y}_i|}{\text{capacity}_i} \times 100\%$
* **Capacity Overrun Rate:** Percentage of test predictions where $\hat{y} > \text{capacity}$.

### 5.2 Held-Out Test Set Results

| Model Architecture | MAE (Tickets) | RMSE (Tickets) | WAPE (%) | Occupancy MAE (%) | Capacity Overrun (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Capacity-Adjusted Baseline** | $36.37$ | $50.47$ | $52.72\%$ | $16.52\%$ | $0.0\%$ |
| **Ridge Regression** | $7.51$ | $11.06$ | $10.89\%$ | $3.46\%$ | $0.0\%$ |
| **Gradient Boosting (Champion)** | **$6.44$** | **$10.35$** | **$9.33\%$** | **$2.80\%$** | **$0.0\%$** |

**Performance Gain:** The Gradient Boosting model reduces MAE by **$82.3\%$** relative to the naive baseline and achieves an occupancy error of under **$3\%$** across the auditorium.

---

## 6. Granular Subgroup & Cold-Start Analysis

Operational evaluation requires verifying model robustness across high-risk and low-data segments:

### 6.1 Cold-Start vs. Mature Releases (Test Set)
* **Cold-Start Shows (Release Days 0–3, $N=80$):**
  * Average Actual Sales: $115.4$ tickets
  * MAE: **$7.62$ tickets** (WAPE: $6.61\%$)
* **Mature Shows (Release Days $>3$, $N=1,014$):**
  * Average Actual Sales: $65.3$ tickets
  * MAE: **$6.34$ tickets** (WAPE: $9.71\%$)

*Analysis:* In cold-start shows, the absolute error is slightly higher ($7.62$ vs. $6.34$ tickets) due to higher variance in opening weekend walk-in demand. However, because opening shows draw larger crowds, the relative error (WAPE) remains tight at $6.61\%$. The model uses star power, genre priors, and $T-24\text{h}$ advance booking pace to bridge the absence of same-movie historical track records.

### 6.2 Screen Tier Performance
| Screen Tier | Capacity | Test Shows | Avg Actual Tickets | MAE (Tickets) | Occupancy MAE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **IMAX** | 420 | 183 | $159.7$ | $14.44$ | $3.44\%$ |
| **Premium Large** | 300 | 183 | $81.4$ | $7.85$ | $2.62\%$ |
| **Standard Mid** | 150–220 | 546 | $47.3$ | $4.55$ | $2.48\%$ |
| **VIP Gold** | 80 | 182 | $30.4$ | $2.64$ | $3.30\%$ |

The Occupancy MAE remains consistent between $2.48\%$ and $3.44\%$ across all hall sizes.

---

## 7. Inference Interface & Operational Decision Support

The solution provides a production-grade FastAPI application with an integrated Web Dashboard:

### 7.1 Key Endpoints
1. `GET /health`: Health status and loaded model metadata.
2. `GET /metrics`: Held-out test evaluation benchmarks.
3. `POST /predict`: Real model inference with $80\%$ confidence intervals ($[\hat{y} - 1.282\sigma,\, \hat{y} + 1.282\sigma]$) and cold-start warnings.
4. `POST /explain`: Interpretable driver breakdown, staffing roster calculation, and screen swap advisory.
5. `POST /predict/batch`: Batch forecasting across multiple halls for schedule optimization.
6. `GET /`: Interactive web dashboard.

### 7.2 Decision-Support Logic
* **Staff Scheduling:** Recommends exact personnel for concessions ($2$ to $7$), floor ushers ($1$ to $4$), and cleaning crew ($1$ to $3$) based on occupancy tiers.
* **Screen Re-allocation:** If projected occupancy exceeds $88\%$, the system flags a "Turnaway / Capacity Saturation Risk" and advises upgrading the show to a larger auditorium (e.g. Screen 1 IMAX). Conversely, if occupancy on a large screen is projected $<25\%$, it suggests downsizing to save energy.

---

## 8. Responsible AI, Bias & Failure Modes

### 8.1 Human-in-the-Loop Governance
The model is strictly designed as a **decision-support tool**. Final screen swaps and staff shift approvals rest with human cinema floor managers. The model does not make automated booking cancellations or high-stakes employment decisions.

### 8.2 Privacy & Sensitive Data
* No Personally Identifiable Information (PII) or individual customer payment data is used.
* All data is aggregate show-level ticketing counts.

### 8.3 Known Failure Modes
1. **Unmodeled External Shocks:** Extreme weather (snowstorms, severe downpours) or sudden transit strikes can suppress walk-in attendance below the $T-24\text{h}$ forecast.
2. **Viral Social Media Spikes:** A sudden breakout viral review between $T-24\text{h}$ and showtime can cause a late walk-in surge beyond the predicted interval.
3. **Off-Calendar Local Events:** Major local concerts or sports finals happening adjacent to the cinema venue may siphon footfall.

---

## 9. Conclusion

The developed solution fulfills all requirements set forth by ALGIONS AI LABS:
* Reproducible end-to-end pipeline with fixed seeds (`RANDOM_SEED = 42`).
* Zero temporal target leakage verified via automated test suites.
* Defensible capacity-bounded modeling delivering **$9.33\%$ WAPE** and **$6.44$ tickets MAE**.
* Tested REST API and responsive Web Dashboard ready for deployment.

