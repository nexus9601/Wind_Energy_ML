# India Wind Energy Intelligence System

**Data-driven spatial siting and predictive operational safety control for wind energy infrastructure in
India using classical machine learning** (Python, scikit-learn, Google Colab; no deep learning).

The project has two machine-learning pipelines and one dashboard:

| | Pipeline A — Site suitability | Pipeline B — Operational safety |
|---|---|---|
| Question | *Where* in India is land suitable for wind farms? | *When* should a turbine brake pre-emptively? |
| Data | NIWE 150 m wind atlas (19.48 M points) + SRTM 1″ terrain + ESA WorldCover 2021 | ERA5 hourly reanalysis 2010–2024 at 12 sites (via Open-Meteo) |
| ML | Feature engineering → PCA → **K-Means** and **DBSCAN** (unsupervised) | **Logistic Regression, SVM, Random Forest** (supervised) + GridSearchCV |
| Output | India suitability map (Low / Moderate / High / Very high) and candidate areas | Next-hour **RUN** / **PRE-EMPTIVE BRAKE** prediction |

Both feed the **Streamlit dashboard** (`dashboard/app.py`).

---

## Folder structure (this shared `data` folder)

```
data/
├── README.md                       ← this file
├── notebooks/                      ← run these in Google Colab, in order
│   ├── 01_niwe_data_cleaning_and_eda.ipynb
│   ├── 02_terrain_landcover_pilot_N22E070.ipynb
│   ├── 03_india_terrain_landcover_integration.ipynb
│   ├── 04_feature_engineering_and_pca.ipynb
│   ├── 05_site_suitability_clustering.ipynb
│   ├── 06_era5_weather_data_download.ipynb
│   ├── 07_turbine_safety_classification.ipynb
│   └── 08_run_dashboard.ipynb
├── dashboard/
│   ├── app.py                      ← Streamlit dashboard
│   ├── requirements.txt
│   └── app_data/                   ← written by notebooks 05 and 07
├── docs/
│   └── Project_Report               ← report (methodology complete; results tables to fill after running)
│
│   (created by the notebooks)
├── raw/                            ← NIWE CSV (downloaded by notebook 03 if missing)
├── terrain/                        ← pilot outputs and SRTM tile inventory (notebook 02)
├── niwe_by_tile/                   ← NIWE split into 400 tile files (notebook 03)
├── integrated/                     ← india_integrated.parquet + tile log + validation report (notebook 03)
├── features/                       ← clustering_features.parquet (notebook 04)
├── era5/                           ← hourly weather per site (notebook 06)
├── models/                         ← feature pipeline, K-Means, safety classifiers (04, 05, 07)
└── results/                        ← one sub-folder per notebook: CSV tables, summary.json, figures/
```

## How to run

1. **Give Colab access to this folder.** It is shared from another account, so in Google Drive open
   *Shared with me* → right-click `data` → *Organize → Add shortcut* → put the shortcut in
   `My Drive/wind-energy-ml/`. Every notebook expects
   `DATA_DIR = /content/drive/MyDrive/wind-energy-ml/data` (change that one line if you put it elsewhere).
2. **Open each notebook in Colab** (double-click in Drive → *Open with Google Colaboratory*) and run all
   cells, in this order:

| # | Notebook | What it does | Approx. runtime | Main outputs |
|---|---|---|---|---|
| 01 | NIWE cleaning & EDA | Data quality on 19.5 M rows, distributions, correlations | done (outputs saved) | — |
| 02 | Terrain & land-cover pilot | Method developed on tile N22E070 | done (outputs saved) | `terrain/` |
| 03 | India terrain & land-cover integration | All 400 tiles, resumable; merge + validation + constraint flags | 1–3 h (resumable) | `integrated/india_integrated.parquet` |
| 04 | Feature engineering & PCA | Constraints, log transforms, redundancy removal, scaling, PCA | ~10 min | `features/`, `models/feature_pipeline.joblib` |
| 05 | Site suitability clustering | K-Means k-selection (scaled vs PCA), profiles, tiers, map, DBSCAN | 20–40 min | suitability map, candidate areas, dashboard files |
| 06 | ERA5 weather download | 12 sites × 15 years hourly, resumable, rate-limit aware | 30–90 min | `era5/era5_hourly_sites.parquet` |
| 07 | Turbine safety classification | Features, next-hour label, 3 models + GridSearchCV, threshold tuning, test & regional evaluation | 30–90 min | models, metrics, dashboard files |
| 08 | Run dashboard | Starts the Streamlit app from Colab | 1 min | — |

Notebooks 06–07 (Pipeline B) do not depend on 03–05, so they can be run in parallel by another team member.

> **Tip:** if Colab disconnects during notebook 03 or 06, just run the notebook again — finished tiles /
> site-years are skipped.

## Key methodological decisions

* **Land cover and slope are constraints, not clustering features.** Points that are mostly water,
  wetland/mangrove, built-up, snow/ice or steep (> 15°) are excluded and shown as a separate category.
  Thresholds are in `CONSTRAINT_RULES` (notebook 03) and should be justified in the report.
* **Land cover per NIWE cell.** NIWE points are centres of a regular 0.004° grid; each cell is exactly
  48 × 48 WorldCover pixels, so class percentages are computed exactly for every point.
* **Redundant features removed by rule** (|r| > 0.95): Weibull A (≈ wind speed) and, if confirmed on the
  real data, air density (≈ function of elevation).
* **PCA is compared, not assumed better:** K-Means is run on both the standardised features and the
  principal components; the silhouette is always measured in the standardised space.
* **Suitability tiers** come from ranking clusters by median wind power density — they are relative
  categories, not an engineering yield assessment.
* **Next-hour prediction**, chronological split (train 2010–2019, validation 2020–2021, test 2022–2024),
  **F2 score** for tuning (recall counts twice), decision threshold tuned for **≥ 90 % recall**.
* **Hazard threshold:** 25 m/s is kept as the *reference* cut-out. Because ERA5 hourly averages over
  ~30 km smooth local peaks, the training threshold is the highest candidate that leaves enough
  hazardous hours (shown in `threshold_selection.csv`).
* **Geographic bias** is measured per site and with leave-one-zone-out retraining.

## Data sources (cite in the report)

* NIWE — National Institute of Wind Energy, *Indian Wind Atlas at 150 m* (wind speed, Weibull A/k, air
  density, wind power density).
* NASA SRTM 1 arc-second DEM (via the AWS Terrain Tiles "skadi" archive).
* Zanaga, D. et al. (2022). *ESA WorldCover 10 m 2021 v200*. doi:10.5281/zenodo.7254221 (CC BY 4.0).
* Hersbach, H. et al. (2020). *The ERA5 global reanalysis*. QJRMS 146:1999–2049; accessed via
  Open-Meteo.com Historical Weather API (CC BY 4.0).
* Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12:2825–2830.

## Status

| Phase | Status |
|---|---|
| 1. NIWE cleaning & EDA | complete |
| 2. Terrain & land-cover integration | pilot complete; nationwide run = notebook 03 |
| 3. Feature engineering & PCA | code ready (notebook 04) |
| 4. K-Means & DBSCAN | code ready (notebook 05) |
| 5. Operational safety classification | code ready (notebooks 06–07) |
| 6. Evaluation & geographic bias | code ready (notebook 07) |
| 7. Dashboard | ready (`dashboard/app.py`, notebook 08) |
| 8. Report & presentation | report draft in `docs/`; fill results after running |

All notebooks 03–08 and the dashboard were tested end to end on synthetic data with the same
structure as the real data; the real-data numbers are produced when you run them.
