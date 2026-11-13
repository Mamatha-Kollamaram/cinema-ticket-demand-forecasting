# 3-5 Minute Video Recording Guide & Presentation Walkthrough

This guide provides a structured script and timing plan for recording your demonstration video.

---

## Video Overview & Target Timing

* **Total Duration:** 3 to 4.5 minutes (meets the 3–5 minute requirement).
* **Suggested Recording Tools:** OBS Studio, Loom, or Windows Snipping Tool / Xbox Game Bar (`Win + Alt + R`).
* **Format:** Screen capture of the interactive web dashboard and terminal with your voiceover.

---

## Step-by-Step Walkthrough Script

### Section 1: Introduction & Objective (0:00 – 0:45)
* **What to Show:** Project title on the web dashboard (`http://127.0.0.1:8000`).
* **Key Talking Points:**
  * Introduce your project: *Cinema Ticket Demand Forecasting*.
  * Define the business problem: Cinema managers need to optimize screen allocations and schedule concession, usher, and cleaning personnel 24 hours prior to showtime ($T-24\text{h}$).
  * State the core technical constraints: The prediction unit is an individual show $(Screen, Showtime, Movie, Date)$. The physical target is tickets sold, bounded strictly between zero and screen capacity ($0 \le \hat{y} \le \text{Capacity}$).
  * Explain the cold-start challenge: Newly released movies (first 3 days) lack same-movie historical ticket history, requiring reliance on prebooking pace, genre baselines, and star power.

---

### Section 2: Data Pipeline & Model Architecture (0:45 – 1:45)
* **What to Show:** Terminal or Jupyter Notebook showing the chronological split and training output.
* **Key Talking Points:**
  * Highlight the chronological validation strategy:
    * Data is split strictly by date (70% Train, 15% Validation, 15% Held-Out Test).
    * Zero temporal lookahead leakage — no future or same-day walk-in data is used.
  * Compare the architectures:
    * **Baseline:** Grouped historical capacity-adjusted occupancy benchmark (Test MAE: 36.37 tickets, WAPE: 52.72%).
    * **Ridge Regression:** Capacity-bounded linear baseline (Test MAE: 7.51 tickets).
    * **Champion Gradient Boosting:** Histogram-based gradient boosted trees with non-linear interaction terms and capacity clipping.
  * Point out the test performance: Test MAE of **6.44 tickets** and **9.33% WAPE**, representing an **82.3% error reduction** over the baseline with 0% capacity overruns.

---

### Section 3: Live Interactive Dashboard Demonstration (1:45 – 3:15)
* **What to Show:** Open `http://127.0.0.1:8000` in your web browser.
* **Actions to Perform on Screen:**
  1. **Demo 1 — Cold-Start Opening Weekend:**
     * Click the button **"IMAX Opening Weekend (Cold-Start)"**.
     * Show that *Days Since Release* is 1 day.
     * Point out the **"Cold-Start Theatrical Window Active"** warning banner.
     * Click **"Generate Capacity-Bounded Forecast"**.
     * Explain the output: Projected tickets sold, expected occupancy percentage, and the 80% confidence interval.
     * Show the **Staff Allocation Card** (concession crew, ushers, cleaning team) automatically calculated based on occupancy.
  2. **Demo 2 — Screen Re-allocation Advisory:**
     * Show the **Screen Utilization Advisor**. If occupancy is high (e.g. >88%), the advisor recommends upgrading to a larger hall to prevent turnaway.
  3. **Demo 3 — Off-Peak Midweek Drama:**
     * Click **"Midweek Indie Drama (Off-Peak)"**.
     * Point out how the forecast updates dynamically for a mature release (Day 18), displaying lower required staffing and recommending an energy-saving smaller screen.

---

### Section 4: Technical Verification & API Walkthrough (3:15 – 4:00)
* **What to Show:** Switch to the browser tab with Swagger documentation at `http://127.0.0.1:8000/docs` or terminal running `pytest`.
* **Key Talking Points:**
  * Show the interactive OpenAPI `/docs` displaying `/predict`, `/explain`, and `/metrics`.
  * Run `python -m pytest -v` in terminal to show all 9 unit and integration tests passing in under 5 seconds.
  * Confirm that the API returns live model outputs with dynamic uncertainty intervals, not hardcoded dummy responses.

---

### Section 5: Governance & Conclusion (4:00 – 4:30)
* **Key Talking Points:**
  * Emphasize responsible AI: The system operates as decision support, preserving human manager discretion for unusual conditions like severe weather or localized disruptions.
  * Conclude by stating that the codebase is modular, fully documented in `README.md` and `REPORT.md`, and ready for deployment.

---

## Live Walkthrough & Debugging Preparation (Q&A Cheat Sheet)

If asked during the technical evaluation to make a live modification or explain decisions:
1. **How is leakage prevented?** 
   * Strict chronological ordering via `chronological_train_val_test_split()`. The training window ends before the validation window starts, which ends before the test window starts. All features are strictly evaluated at $T-24\text{h}$.
2. **How are physical capacity bounds enforced?**
   * The `CapacityBoundedRegressor` class enforces `np.clip(raw_preds, 0, screen_capacity)`. Predictions can never exceed auditorium seat limits.
3. **How does the model handle cold-start movies?**
   * Identified by `is_cold_start = (days_since_release <= 3)`. The model leverages genre priors, star power ratings, and early advance prebooking pace ($T-24\text{h}$), while expanding prediction uncertainty intervals.
4. **How to run the server:**
   ```bash
   uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
   ```

