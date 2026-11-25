# Changelog

All notable changes to FireScope are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The authoritative working notes live in
[`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md).

## [3.3-stable] — 2026-11-25

Training-data integrity, failure alerting, and security hardening.

### Added
- **Structure-damage archive** (`backend/data/dins/`, `data/dins_archive.py`): every CAL FIRE DINS
  fire-damaged structure 2013 → latest published (77,515 across 2013–2025; CAL FIRE's point-level
  inspections begin 2013-08-07), validated (fire hazard, recorded damage, in-California coordinates,
  incident name/number/date, de-duplicated) and checked against CAL FIRE's own counts per year. Each
  incident keeps its full inspection tally, including undamaged and inaccessible structures.
  `dins-sync.yml` refreshes the current and previous year daily; `/api/history/dins` serves the
  archive and falls back to a live, identically filtered fetch.
- **Three-source, skill-weighted weather** for training rows: NOAA RTMA (2.5 km hourly analysis of
  surface observations, via Earth Engine) joins ERA5 and MERRA-2. Per-feature weights come from measured
  error against 212 station samples (15 ASOS airports + 7 independent USCRN sites, Jun–Nov 2026),
  following the NWS National Blend of Models: humidity is blended (RTMA 0.55 / ERA5 0.33 / MERRA-2
  0.11); temperature and wind use RTMA with ERA5 then MERRA-2 as fallbacks. Error vs stations: humidity
  4.47 RH pts (ERA5 alone 6.20), temperature 0.59 °C (1.14), wind 1.98 m/s (3.34). A source that grossly
  disagrees with the other two is dropped for that feature; two disagreeing sources still reject the row.
- **Weather App Plan data path**: daily ingest and the offline backfill (`ml/backfill_matched.py`) pick
  fires with one shared function (one detection per ~5 km cell per day, seeded shuffle), screen static
  heat sources (refineries, plants; 168 cells from FIRMS archive type-2 detections), and keep complete
  fire/control pairs only. Backfilled rows use FIRMS archive vegetation fires only (type 0).
- **Weather App Plan dataset** (`training_data/weather_app_plan/california_weatherapp.csv`): backfill
  2018 → present plus re-graded rows; daily ingests are appended to it and to the original matched set.
  Original training files are unchanged.
- **Weekly training-data quality gate** (`POST /api/internal/ml/weather-skill`, `ml/weather_skill.py`):
  scores RTMA, ERA5, MERRA-2 and the production blend against 22 stations, flags degradation, lost
  sources, weight drift and unpaired recent fires, emails the details and fails the weekly run, which
  blocks promotion.
- **Matched-control training set** (`training_data/california_matched.csv`): every fire is paired with
  the same point on the same date one year earlier, with every time-varying feature backdated to that
  date. Replaces uniform-random no-fire points, which taught the model *where* fires happen (and
  inverted the humidity/temperature/wind priors) rather than fire *conditions*.
- **Backup source for every training feature**, each measuring the same quantity: weather Open-Meteo
  archive (ERA5) → NASA POWER (MERRA-2); EVI Earth Engine → NASA ORNL DAAC (same MOD13Q1 composite
  rule); KBDI spinup NASA POWER → Open-Meteo archive; elevation USGS 3DEP → Open-Elevation. A point with
  no source left is dropped, never interpolated.
- **Weather cross-reference** between two independent reanalyses (ERA5 vs MERRA-2, same ±3-day window
  and definitions); tolerances (8 °C / 30 RH pts / 5 m/s) sit above the largest disagreement measured
  on validated rows.
- **Failure emails to the owner**: an hourly digest of every failure, fallback and error the API logs
  (`services/failure_digest.py`); `notify-failure.yml` emails the failed jobs/steps and log tail of any
  workflow run that doesn't succeed; the daily ingest alerts when a run returns only one class.
- Date-targeted ingest backfill (`daily-retrain.yml` `date` input); per-point fetch failures, class
  counts and single-source counts in the ingest response and workflow log.
- Per-IP rate limits on `/login` (plus per account), `/register` and `/predict*`.

### Changed
- Daily ingest uses detections from **6 days back** so every feature is observed, not forecast, and
  present in both weather sources.
- Controls pair only with accepted fires; even 30 s / 30 s fire/control budget.
- Weekly gate: leak-free, location-grouped holdout (the previous holdout let the production model be
  scored on rows it was trained on, so every candidate was rejected from 2026-08-24 to 2026-11-16).
- **Weekly auto-promotion held in dry-run** until the first matched-control model is reviewed.
- gunicorn runs with `--threads 8`, so a long ingest no longer stalls page loads.
- Training-data stall watchdog fails the run red after 2 days without new data.

### Fixed
- **History showed ~30% of recorded fires**: `/api/history/perimeters` defaulted to a 100-acre floor,
  hiding 10,453 of CAL FIRE's 23,334 perimeters (2025: 113 of 521), including small fires that destroyed
  homes. It now returns every recorded fire; with all sizes shown, 99.82% of fire-damaged structures link
  to a FRAP fire by name, complex name or location (the other 140 belong to fires CAL FIRE never mapped).
- **GNews rate limit**: GNews (100 requests/day) was called on every 15-minute news refresh and after every
  restart. It is now called at most hourly and backs off an hour after a failure.
- **History → structure damage**: 41% of mapped points were undamaged inspections; the map now shows
  only fire-damaged structures. The popup read an address field the API never requested (always
  "Unknown address"); it now shows the address, damage level, the fire that caused it, that fire's
  cause, and (where recorded) how fire reached the structure, hiding unrecorded fields. The two
  point-size/opacity sliders were removed.
- **Database out-of-memory crashes**: the 256 MB Postgres was OOM-killed 13 times (2026-11-20 and -25).
  The boot cache warm no longer loads the multi-MB fire-history rows, and past-year fire history never
  expires, so it is never recomputed and rewritten.
- Manual (backfill) ingest runs no longer redeploy the API once per day processed; only the scheduled run
  and promote runs deploy.
- County risk: a failed weather batch no longer fans out into repeated Open-Meteo calls per county.
- Failure digest subject reads "since the last report" (the shutdown flush is not hourly).
- A data push rejected because `main` moved during the run dropped that run's rows; the push now
  rebases and retries.
- **Daily ingest stopped on 2026-11-06**: the NASA FIRMS S-NPP NRT feed went empty; switched to
  VIIRS NOAA-20.
- Ingest rows could silently carry interpolated EVI/KBDI when a live source failed; 11 such rows from
  2026-11-20 removed.
- The previous second weather source compared a current-hour reading with a 3-day mean and dropped
  valid fire rows.
- Archive-weather lookups failed for the last 3 days of dates (end date past today); now clamped and
  retried.
- The zone-risk fallback multiplied one failed weather batch into many Open-Meteo calls, exhausting
  the shared rate limit for other callers.
- `quarantine.csv` was gitignored, so rejected-row records were never kept.
- Daily pre-warm failed every day on the retired Cal OES shelters source, which also stopped the
  endpoints after it from warming.

### Security
- The failure digest redacts credentials embedded in upstream error URLs (query keys and path keys) before
  emailing; a library retry still in progress is no longer reported as a failure.
- Removed a legacy, unwired-in-docs alert blueprint whose routes lacked authentication, and its
  scheduler (its jobs also failed on every run).
- Admin cron endpoints now require the internal token.
- Signup never grants the Admin role.
- Production refuses to start without `SECRET_KEY` and `JWT_SECRET_KEY`.
- Removed an unused component that embedded an API key.

### Removed
- `backend/data/weather_crosscheck.py` (MET Norway cross-check), `backend/routes/alerts.py`,
  `frontend/src/components/FIRMSMap.tsx`.

## [3.2-stable] — 2026-06-19

### Added
- **Data-quality gate on every ingested row** (`routes/ml_ingest.py`): physical-range/sanity checks,
  a low-confidence FIRMS label filter, no-fire verification (reject points inside an active NIFC
  perimeter or near a fire in a 3-day window), a California-land mask on no-fire sampling, and
  **cross-source weather corroboration** (Open-Meteo vs MET Norway — drop grossly-disagreeing
  points). Rejected rows are quarantined to `training_data/quarantine.csv`.
- **Weekly data-quality monitors**: outlier rate (robust z) + distribution drift (PSI, recent vs
  earlier ingest) via `POST /api/internal/ml/data-health` (weekly digest with a random row sample);
  `POST /api/internal/ml/backtest` scores recent real fires against the live model (alerts on low
  recall); `POST /api/internal/ml/feature-audit` cross-validates cached elevation vs an independent
  DEM. Email alerts via `POST /api/internal/ml/alert` (feed-outage detection in the daily run too).
- **Weekly gated auto-promotion** (`.github/workflows/weekly-promote.yml`, Sunday night): retrains on
  the full rolling dataset and promotes only if the candidate clears the physics + AUROC/Brier gate;
  archives the previous model, deploys, and emails the owner (`POST /api/internal/ml/promotion-email`).
  Manual promotion any day before Sunday is preserved.
- About page (Settings → About): documented the model, scoring, live inputs, data-quality
  safeguards, and the weekly monitoring + auto-promotion.
- **Constant/dead-feature detector** (`ml/data_quality.constant_features`): flags any feature whose
  values are ≥95% identical in the recent window (a flatlined feature that, being in-range, evades
  the outlier monitor). Surfaced in the data-health report + email.

### Fixed
- **Dead wind feature**: `_features_for` read the wrong key (`wind` vs Open-Meteo's `wind_speed`), so
  wind was recorded as `0.0` on every ingested row. Now records real wind (ingest path only). A
  **wind sanity check** in `retrain_and_gate._load_dataset` also excludes the historical zero-wind
  rows (`wind <= 0`) from the training/evaluation union so they cannot degrade auto-promotion.
- **Navbar stacking**: header z-index is set inline (this build doesn't compile Tailwind arbitrary
  `z-[…]`), and `<main>` is isolated, so the sticky nav stays above all map overlays on every page.
- Removed stale "random forest" wording from current model descriptions, GitHub docs, and the
  website (the live model is the monotonic HistGradientBoosting + isotonic classifier; version
  history retains its original wording).

### Changed
- Shelters & Evacuation: surface CalOES shelter-in-place orders; "Show on map" buttons render only
  when there's something to show; directions live only on open-shelter cards (not evacuation zones).
- Ingest point-loop budget 55s → 35s for headroom (added perimeter load + wider FIRMS fetch +
  per-point cross-source call).

## [3.1-stable] — 2026-06-03

### Added
- **Monotonic risk model (promoted to production)** — `HistGradientBoostingClassifier` with
  monotonic constraints `[0,1,1,-1,0,1]` over `[evi, air_temp, wind, humidity, elevation, kbdi]`,
  isotonic-calibrated, behind a **PDP physical-direction gate** that rejects any candidate whose
  constrained features point the wrong way. Replaces the CalibratedRF that had learned backwards
  relationships (wind anti-correlated, humidity inverted, KBDI saturating, dead EVI slider).
- **Continuous daily retraining, entirely in GitHub** — dataset committed in-repo (zero DB
  cost-risk); `daily-retrain.yml` ingests FIRMS detections → appends → gated retrain → promote.
- **Per-user saved zone overrides** (researcher page) — `user_overrides` table, 24h TTL,
  "Save for 24 hours" button, reset this/all zones, 20-zone shared total cap across all 4 zone
  types, EVI slider fixed to 0–1.
- **Active-fire accuracy** — `/api/fire-perimeters` shows only real active fires (WF type, not out,
  <100% contained, current ≤14 days): 77 → 7. Enriched fire popups; active-fire totals on the
  dashboard and Shelters & Evac.

### Changed
- Evacuation header splits **orders vs warnings**; 7-day active-only filter drops stale/`**TEST**`
  zones; real cause surfaced via `NOTES`/`EDIT_DATE`. Shelters are OPEN-only with a 5-min refetch.

### Fixed
- **Cache durability** — root-caused a 13-day stale-feed freeze: `endpoint_cache.computed_at` froze
  at first insert. Timestamps now advance on every write; dirty transactions roll back before save;
  a force-recompute backstop logs any stale row.
- **Deploy reliability** — `restart-after-backend-deploy.yml` calls the Render restart API because a
  "live" deploy does not reliably restart gunicorn (old workers served stale code).
- **Typecheck gate** — added `frontend/tsconfig.json` + `scripts/typecheck-gate.cjs` to `npm run
  build`, blocking the undefined-reference crash class that white-screened the app post-login.

## [3.0-stable]

### Added
- **Fourth alert channel — wildfires in your county** — per-county bundled CAL FIRE incident alerts
  every 10 min, change-driven SHA-256 dedup, one-final-"fully contained" email then silence.
- **Universal one-click email unsubscribe** — RFC 8058 `List-Unsubscribe` native Gmail/Apple button,
  per-channel footer links, HMAC-token-authed public endpoint.
- **Breaking-news pipeline cleanup** — NWS title cleaning, per-user CA-county filter, two-pass
  cross-source dedupe, live CAL FIRE containment enrichment.

### Changed
- Fire-perimeter map auto-retires contained fires; high-risk alerts capped at 20 locations.
- Email shell hardened to be Gmail-bulletproof (table-based, hex colors, hosted PNG markers).

## [2.9-stable]

### Changed
- **Canonical NFDRS 5-tier risk model** — replaced the custom 9-tier scale with the standard National
  Fire Danger Rating System bands (Low / Moderate / High / Very High / Extreme), wired through a
  single source of truth across every polygon overlay and the email renderer.
### Added
- **Shelters & Evacuation page rebuild** — ref-based overlay (no flicker), OPEN-only shelters,
  independent shelter/zone/fire popups, status banners with fit-to-bounds.
- **Researcher shelter overlay** toggle; **county-match + pickable-fire + shelter-opened** alert
  improvements; evac cron moved to in-process cache reads (20–90 s → 0.5–5 s).

## [2.8-stable] — and earlier

- Four alert channels and the Resend email pipeline; 5-tier risk palette; map and history surfaces.
- **v2.7 / v2.6 model evolution** — moved from KNN/uncalibrated RF to a sigmoid-calibrated Random
  Forest over 6 features (adds **KBDI** drought index), real MODIS EVI via Google Earth Engine +
  USGS elevation, date-restratified negatives, spatial-block cross-validation, and SHAP attribution.
  Headline: 91.7% accuracy / ROC-AUC 0.964 on the spatial-CV regime.

See [`docs/SESSION_HANDOFF.md`](docs/SESSION_HANDOFF.md) for the full per-release detail.
