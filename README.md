# Anticipatory Solar Irradiance Forecasting

**A two-layer framework that forecasts solar irradiance *and* warns 3 hours in advance when that forecast is likely to be wrong.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B)
![SHAP](https://img.shields.io/badge/SHAP-explainability-8A2BE2)

Most solar forecasting models predict how much sunlight there will be. Grid operators also need to know **when to stop trusting the forecast**, for example before a dust storm. This project adds a second model that predicts high-error windows ahead of time so reserves can be prepared.

```
LAYER 1: Base forecaster
  Weather features → Random Forest → Global Horizontal Irradiance (W/m²)

LAYER 2: Meta-error forecaster
  Atmospheric trends + recent errors → LightGBM (SMOTE + tuned threshold)
  → "High forecast error expected at T+3h" alert
```

## Results (test year: 2024)

**Layer 1: irradiance forecast (Random Forest)**

| City | RMSE (W/m²) | R² |
|---|---|---|
| Riyadh | 51.68 | 0.975 |
| Jeddah | 49.49 | 0.977 |
| Abha | 67.70 | 0.956 |
| Dammam | 49.19 | 0.976 |
| Qassim | 47.28 | 0.979 |

**Layer 2: 3-hour-ahead high-error alerts (LightGBM)**

| City | High-error windows | Caught | Recall | Precision |
|---|---|---|---|---|
| Riyadh | 289 | 194 | 67.1% | 35.0% |
| Jeddah | 313 | 154 | 49.2% | 28.2% |
| Abha | 641 | 499 | 77.8% | 31.5% |
| Dammam | 293 | 154 | 52.6% | 37.4% |
| Qassim | 288 | 172 | 59.7% | 29.7% |

Recall is prioritized over precision: a false alarm costs a manual check, while a missed alert can mean an unexpected drop in solar output.

<p align="center">
  <img src="colab_all_figures/figures/fig3_performance_comparison.png" width="48%">
  <img src="colab_all_figures/figures/fig10_alert_heatmap.png" width="48%">
</p>

## Data

- **Source:** [Open-Meteo Historical Weather API](https://open-meteo.com/) (CC BY 4.0)
- **Coverage:** hourly, 2019–2024, five Saudi cities with different climates: Riyadh (desert), Jeddah and Dammam (coastal), Abha (mountain), Qassim (semi-arid)
- **Split:** train 2019–2022, validation 2023, test 2024 (temporal, no leakage)

## Method

**Feature engineering:** besides weather variables and interaction terms (`hour × cloud`, `temp × humidity`, clear-sky ratio), the model adds:
- `dust_proxy` = wind speed × (1 − humidity/100): strong, dry wind signals dust
- circular wind direction encoding (`sin` / `cos`)
- 1-hour changes and 3-hour trends for cloud cover, wind, humidity and irradiance
- lagged residuals of the Layer 1 model (recent error history)

**Layer 2 target:** will Layer 1's absolute error exceed 100 W/m² at hour T+3?
Alert hours are only about 3% of the data, so the model combines **SMOTE**, class weights and a **city-specific decision threshold** chosen from the precision-recall curve.

## Repository structure

```
├── 01_data_collection.ipynb  # download 2019–2024 hourly weather from Open-Meteo
├── 02_preprocessing.ipynb    # cleaning + feature engineering
├── 03_baseline_model.ipynb   # Layer 1: Random Forest per city
├── 04_meta_model.ipynb       # Layer 2: LightGBM meta-error model
├── requirements.txt
├── data/
│   ├── raw/                  # hourly weather per city (Open-Meteo)
│   └── processed/            # cleaned data + engineered features
├── models/
│   ├── *_baseline_rf.pkl     # Layer 1 Random Forest per city
│   ├── meta/                 # Layer 2 v1 (LightGBM)
│   └── meta_improved/        # Layer 2 final (LightGBM + SMOTE + threshold)
├── results/
│   ├── baseline/             # Layer 1 residuals
│   ├── meta/                 # Layer 2 v1 metrics
│   └── meta_improved/        # per-hour forecasts, alert probabilities + seasonal predictions
└── colab_all_figures/        # notebooks and PNGs for figures 1–10
```

## Quick start

```bash
git clone https://github.com/redajelfa13/solar-irradiance-forecasting.git
cd solar-irradiance-forecasting
pip install -r requirements.txt
```

**Reproduce the pipeline:** run the notebooks in order (`01 → 02 → 03 → 04`). Each one reads the previous step's output from `data/`, `models/` or `results/`. The data and trained models are already included, so you can also start at any step.

**Or load the trained models directly:**

```python
import joblib, pandas as pd

rf   = joblib.load("models/riyadh_baseline_rf.pkl")
meta = joblib.load("models/meta_improved/riyadh_improved_lgbm.pkl")
preds = pd.read_csv("results/meta_improved/riyadh_improved_results.csv")
print(preds[["timestamp", "GHI", "GHI_predicted", "alert_prob", "predicted_alert", "actual_alert"]].head())
```

## Authors

- Marwa Hanini
- Reda Jelfa, [LinkedIn](https://linkedin.com/in/reda-jelfa)

This work extends Aldossary & Alameen (2026), *"Bridging accuracy and interpretability in solar forecasting: A regional ensemble learning approach"*, Energy Reports. It replaces 30 days of spring data with five years of all-season data, adds dust-related features, and turns a reactive error flag into a 3-hour-ahead alert.
