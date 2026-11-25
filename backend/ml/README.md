# ML Module

Wildfire risk model integrated into the prediction API. The model loads once on the first request and is cached in memory for all subsequent calls.

## How it works

`inference.py` exposes a single function:

```python
predict_from_features(evi, air_temp_encoded, wind, humidity, elevation, kbdi) -> dict
```

The prediction route (`routes/predict.py`) calls this with live data fetched per-request:

| Feature | Live source | Fallback |
|---|---|---|
| `evi` | NASA AppEEARS (spring composite, ~May 1) | nearest sample location |
| `air_temp_encoded` | Open-Meteo air temperature, encoded | nearest sample location |
| `wind` | Open-Meteo current wind speed | nearest sample location |
| `humidity` | Open-Meteo relative humidity | nearest sample location |
| `elevation` | Open-Elevation API | nearest sample location |
| `kbdi` | NASA POWER 30-day window via `data/live_kbdi.py` | 200.0 (mid-range) |

The nearest fallback location is selected using great-circle (haversine) distance, not Euclidean degrees.

## Model details

| Property | Value |
|---|---|
| Algorithm | Monotonicity-constrained `HistGradientBoostingClassifier`, isotonic-calibrated |
| Trainer | `train_monotonic.py` (model) + `retrain_and_gate.py` (gated promotion) |
| File | `models/wildfire_model_predictive.pkl` |
| Training data | `training_data/california_matched.csv`: every fire paired with a same-point control one year earlier, grown daily by the ingest cron, de-duped. (`california_daily.csv` is the historical record of the earlier random-point sampling and is no longer trained on.) |
| Evaluation | 80/20 stratified held-out split; promotion gated on physics (monotonicity) + AUROC/Brier non-regression |
| Metrics | See `models/RETRAIN_LOG.md` and `models/model_metadata.json` (last promotion 2026-06-03: held-out AUROC ≈ 0.888) |
| Features | EVI, air temperature (encoded), wind speed, humidity, elevation, KBDI |
| Output | Risk probability (0–1) + 5-tier NFDRS label (Low / Moderate / High / Very High / Extreme) |

The model is trained with **monotonic constraints** (`monotonic_cst=[0,1,1,-1,0,1]` for EVI, air temp, wind, humidity, elevation, KBDI) so risk can only rise with temperature, wind, and drought and fall with humidity — it can never learn a physically-backwards relationship. It is then wrapped in `CalibratedClassifierCV` with **isotonic** regression (`cv=5`) so `predict_proba` returns calibrated probabilities; the 5-tier thresholds in `inference.py` correspond to empirical fire frequencies rather than raw scores.

**Promotion gate.** Before any retrained candidate replaces the live model it must pass two checks (`retrain_and_gate.py`): (1) a hard PDP physical-direction check (`validate_monotonicity` sweeps each constrained feature across its range and requires it to move the correct direction), then (2) held-out AUROC and Brier must not regress versus the current model. A model that fails physics is never shipped, even if its raw accuracy looks higher.

## Where the training data lives (`ml/training_data/`)

| Location | What | Rows |
|---|---|---|
| `california_2020_kbdi.csv` | **Original** 2020 base set | 1,350 |
| `california_daily.csv` | **Original** daily ingests, Jun–Nov 2026 (random no-fire points; not trained on) | 1,401 |
| `california_matched.csv` | **Original** matched-control set: 2020 fires (2019 controls) + daily ingests from Nov 2026 | grows daily |
| `weather_app_plan/california_weatherapp.csv` | **Weather App Plan** set: backfilled 2018 → present and re-graded rows, all built with the 3-source skill-weighted weather; daily ingests append here too | grows daily |
| `quarantine.csv` | Rows rejected by the ingest gate, with reasons | — |
| `static_heat_sources.csv` | 168 static heat-source cells (refineries, plants) screened from NRT fire labels | — |

Every original file stays at the top level, unchanged; the Weather App Plan data has its own folder.
Daily ingests are appended to both `california_matched.csv` and the Weather App Plan set.

## Data quality & continuous training

The training set grows daily and the model self-updates weekly, both fully in GitHub Actions (no DB
cost-risk — the dataset lives in-repo). Every row is vetted before it can train the model.

**Ingest (per row, `routes/ml_ingest.py`):**
- **Observed data only** — the daily run ingests detections from 6 days back (`INGEST_LAG_DAYS`), so
  the whole ±3-day weather window is observed and present in both weather sources.
- **Matched controls** — each accepted fire gets a control at the same lat/lon one year earlier, with
  every time-varying feature backdated to that date. Controls pair only with accepted fires.
- **Weather: three sources, skill-weighted** — NOAA RTMA (2.5 km hourly analysis of surface
  observations, via Earth Engine), Open-Meteo archive (ERA5) and NASA POWER (MERRA-2), fetched in
  parallel. Weights per feature come from measured error against 212 station samples (15 ASOS + 7
  independent USCRN sites, Jun–Nov 2026, leave-one-site-out): humidity RTMA 0.55 / ERA5 0.33 /
  MERRA-2 0.11; temperature and wind RTMA, falling back to ERA5 then MERRA-2 (`WEATHER_WEIGHTS` in
  `routes/ml_ingest.py`). Mean error vs stations: humidity 4.47 RH pts (ERA5 alone 6.20), temperature
  0.59 °C (1.14), wind 1.98 m/s (3.34).
- **Backup per feature (same quantity)** — weather: any of the three sources;
  EVI: Earth Engine → ORNL DAAC (same MOD13Q1 composite rule); KBDI spinup: NASA POWER → Open-Meteo
  archive; elevation: USGS 3DEP → Open-Elevation. No source left → the point is dropped, never
  interpolated.
- **Cross-reference** — with three sources, one more than 8 °C / 30 RH pts / 5 m/s from the median is
  dropped for that feature and the row kept; with only two that disagree there is no tiebreak, so the
  row is dropped.
- **Range/sanity, label confidence, CA-land mask** as before.
- Rejected rows are appended to `training_data/quarantine.csv` (committed) for review; per-point
  fetch failures are returned in the ingest response and printed in the workflow log.

**Weekly monitors (`POST /api/internal/ml/...`):**
- `data-health` — robust-z outlier rate + PSI drift (recent vs earlier ingest) + a random row sample.
- `backtest` — scores recent *real* fires with the live model; alerts if recall@High < 0.5.
- `feature-audit` — cross-validates cached elevation against an independent DEM.
- All email the owner on a problem (`alert`); the weekly run sends a digest regardless.

**Promotion:**
- `daily-retrain.yml` (daily) ingests + retrains a candidate **dry-run** (logs the gate decision).
- `weekly-promote.yml` (Sunday night) retrains and evaluates the gate (physics + AUROC/Brier
  non-regression on a leak-free holdout). **Currently held in dry-run** (`--dry-run`) until the first
  matched-control model is reviewed; remove the flag to re-enable auto-promotion.
- Manual promotion any day: run `daily-retrain.yml` with `promote=true`.

## Feature encoding

| Feature | Encoding | Notes |
|---|---|---|
| `evi` | Scaled float (0–1) | Spring EVI composite (May 1 target); MODIS scale factor 0.0001 applied |
| `air_temp_encoded` | `(T_celsius + 273.15) / 0.02` | Air temperature from Open-Meteo |
| `wind` | m/s (float) | Open-Meteo 10m wind speed |
| `humidity` | % (0–100) | Open-Meteo relative humidity at 2m |
| `elevation` | meters (float) | Open-Elevation terrain height |
| `kbdi` | Float (0–800) | Keetch–Byram Drought Index — cumulative deep-soil moisture deficit over the prior 30 days, in units of 0.01 inch |

## Training pipeline

`build_dataset.py` builds the base training CSV from scratch:

1. **Fire points** — fetched from NASA FIRMS area API (MODIS_SP, California bounding box, 2020)
2. **No-fire points** — randomly sampled within California, at least 50 km from any fire detection (haversine), assigned random 2020 dates to avoid seasonal bias
3. **EVI** — fetched via NASA AppEEARS batch API; spring composite (closest to May 1) used as pre-season fuel load indicator; task ID saved to `.appeears_task_id` to allow resume on network failure
4. **Weather** — historical Open-Meteo data per point (wind, temperature, humidity); 4-attempt retry with backoff
5. **Elevation** — Open-Elevation per point

`enrich_kbdi.py` adds the KBDI column:

6. **R climatology** — `build_r_cache.py` precomputes per-cell mean annual precipitation from NASA POWER and saves it to `r_cache.json`
7. **KBDI** — for each row, fetch the 30-day daily weather window preceding `acq_date` from NASA POWER, look up R from the cache, then run `kbdi.py` (pure-math implementation of Keetch–Byram 1968) to compute the index. Resume-safe — already-enriched rows are skipped on re-run.

`restratify_dates.py` rebalances the no-fire date distribution so the model can't trivially learn "summer = fire."

`retrain.py` trains the calibrated Random Forest, runs both random and spatial CV, saves model + scaler, and auto-generates charts and `charts/RESULTS.md`.

## Files

```
ml/
├── inference.py          # predict_from_features() — loads model, scales features, returns score
├── build_dataset.py      # Base training pipeline (FIRMS + AppEEARS + Open-Meteo + Open-Elevation)
├── kbdi.py               # Pure-math Keetch–Byram Drought Index (Keetch & Byram 1968)
├── build_r_cache.py      # Precompute per-cell mean annual precipitation from NASA POWER
├── enrich_kbdi.py        # Add KBDI column to the training CSV using r_cache + POWER 30-day windows
├── restratify_dates.py   # Rebalance no-fire date distribution to remove seasonal bias
├── retrain.py            # Calibrated RF training, random + spatial CV, chart + summary generation
├── charts/
│   ├── RESULTS.md             # Auto-generated metrics summary
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── metrics_bar.png
│   ├── feature_distributions.png
│   └── calibration_curve.png  # Predicted vs observed fire frequency, before/after sigmoid calibration
└── models/
    ├── wildfire_model_predictive.pkl    # Current live model: monotonic HGB + isotonic calibration
    ├── wildfire_scaler_predictive.pkl   # Fitted StandardScaler
    └── model_metadata.json              # Feature column order + train timestamp
```
