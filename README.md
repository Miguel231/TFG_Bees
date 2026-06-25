# TFG — Early Detection of Swarming Events in Honey Bee Colonies

**Author:** Miguel Arpa Robig  
**Supervisor:** Vicenç Soler Ruíz (Dept. Microelectrònica i Sistemes Electrònics, UAB)  
**Degree:** Final Degree Project — Artificial Intelligence, Escola d'Enginyeria (UAB)  
**Academic year:** 2025/26

---

## Overview

Multi-sensor IoT data from the UAB campus apiary across three seasons (2023–2026), 8 instrumented hives, 23 confirmed swarming events. The project develops machine learning models for early swarm prediction and a secondary honey super placement model, deployed as a local FastAPI + Streamlit production system.

**Contributions:**
- Two biologically interpretable pre-swarm signatures identified: weight–temperature Pearson correlation breakdown (normal r = +0.32 → pre-swarm r = −0.34, p < 0.001) starting ~7 days before departure, and nocturnal temperature increases of +2–+8 °C in the pre-swarm week
- `weight_std_roll14` (14-day rolling std of daily hive weight) confirmed via ablation as the only engineered addition that consistently improves both XGBoost and LSTM across all model variants
- Four model variants with progressively richer features: daily aggregates (NB01, 24 features), morning-window 10–14h (NB02, 31), night-enhanced 00–05h (NB03, 44), honey super placement (NB04, 19–52)
- Walk-forward temporal cross-validation with G-Mean threshold optimisation; operational evaluation via per-season swarm detection count and false alarm count
- XGBoost and LSTM (unidirectional and bidirectional) compared across all configurations with Optuna hyperparameter search

**Best deployed swarm model:** NB01 LSTM Uni (3d) — G-Mean = 0.866, 13/14 test events, 136 false alarms.  
NB01 was chosen over NB03 (higher AUC = 0.882, but 12/14 events) because its 24 daily features are simpler to compute and more robust to sensor downtime.

---

## Repository Structure

```
TFG_Bees/
├── notebooks/
│   ├── exploratory/                       # Preliminary behavioural studies
│   │   ├── 01_eda_seasonal.ipynb              Seasonal weight cycles, intraday foraging patterns
│   │   ├── 02_instability_index.ipynb         Rolling multi-sensor instability index
│   │   ├── 03_nocturnal_signals.ipynb         Nocturnal signal validation vs Ferrari et al. (2008)
│   │   └── 04_external_disturbance.ipynb      18/04/2026 UAB festival cross-hive anomaly
│   └── models/
│       ├── 01_swarm_multihorizon.ipynb        XGBoost + LSTM at 3/7/14-day horizons (24 features)
│       ├── 02_swarm_morning_window.ipynb      Morning-window 10–14h features + weight_std_roll14 (31)
│       ├── 03_swarm_night_enhanced.ipynb      + nocturnal 00–05h aggregations (44 features)
│       ├── 04_honey_super.ipynb               Honey super placement (LightGBM/XGBoost, 19–52 features)
│       └── optuna_nb01.py                     Standalone Optuna search for NB01 LSTM
├── api/                                   # Deployed prediction system
│   ├── main.py                                FastAPI backend (/predict, /weekly-run, /weekly-alza-update)
│   ├── dashboard.py                           Streamlit dashboard (6 tabs)
│   ├── inference.py                           LSTM inference pipeline
│   ├── data_fetcher.py                        Playwright scraper — xlsx download from beehivemonitoring.com
│   ├── update_alza_features.py                Weekly alza feature CSV updater
│   ├── train_alza_features.py                 Honey super model grid search and selection
│   ├── models/                                Serialised model artefacts (committed)
│   │   ├── lstm_uni_01_3d.pt                  Swarm LSTM Uni — 3d horizon (NB01)
│   │   ├── lstm_bi_01_14d.pt                  Swarm LSTM Bi — 14d horizon (NB01)
│   │   ├── scaler_01.pkl                      StandardScaler fitted on NB01 training split
│   │   ├── median_fill_01.json                Training-split medians for NaN imputation
│   │   ├── model_meta_01_3d.json              Architecture params for the 3d model
│   │   ├── model_meta_01_14d.json             Architecture params for the 14d model
│   │   ├── lgbm_alza_solo_hive_7d.pkl         Honey super model — 7d horizon
│   │   ├── lgbm_alza_solo_hive_14d.pkl        Honey super model — 14d horizon
│   │   └── lgbm_alza_meta.json                Alza model metadata (feature lists, thresholds)
│   └── n8n_workflow.json                      n8n automation (weekly Monday 9am)
├── docs/
│   └── images/                            # Result figures referenced in this README
├── data/                                  # NOT in git — place CSVs here (see Data Setup)
│   └── .gitkeep
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Data Setup

Raw sensor data is not included in this repository (files are 150–220 MB).

Place the unified CSV in `data/`:

```
data/
└── 12062026all_boxes.csv     # Full dataset: all hives, Jan 2023 – Jun 2026
```

The actual CSVs live at repo-root level under `../csv/`. The notebooks and `update_alza_features.py` resolve the path automatically.

`01_swarm_multihorizon.ipynb` additionally needs a preprocessed daily features file:

```
data/
└── daily_data.csv            # Pre-computed daily aggregates (from 01_eda_seasonal.ipynb)
```

The honey super notebook and the API use:

```
daily_features_final.csv      # Daily feature matrix for all 17 alza hives (notebook training)
daily_features_prod.csv       # Same structure, updated weekly by /weekly-alza-update (API)
```

**CSV sensor columns:** `Hive name`, `Time`, `Weight` (kg), `Frequency` (Hz), `Volume`, `Temperature heart` (°C), `Humidity heart` (%), `Temperature scale` (°C), `Humidity scale` (%).

Two interleaved sensor nodes per hive (acoustic+internal and scale+external) log at a combined ~10-minute cycle. The preprocessing pipeline merges them into a regular 15-minute grid; gaps ≤ 2 h are forward-filled (`ffill(limit=8)`); longer gaps remain as NaN.

---

## Notebooks Guide

### Exploratory (run first)

| Notebook | Description | Key output |
|---|---|---|
| `01_eda_seasonal.ipynb` | Seasonal weight cycles, intraday foraging by season, inter-hive strategy differences | `daily_data.csv` |
| `02_instability_index.ipynb` | Rolling variability index combining temp, acoustic, weight | Instability time-series around events |
| `03_nocturnal_signals.ipynb` | Nocturnal freq/temp vs Ferrari et al. (2008); weight–temperature correlation breakdown | Comparison figures |
| `04_external_disturbance.ipynb` | 18/04/2026 UAB festival anomaly — cross-hive cross-correlation distinguishes external disturbances from pre-swarm signals | Cross-correlation plots |

### Models

| Notebook | Input | Features | Horizon | Best model |
|---|---|---|---|---|
| `01_swarm_multihorizon.ipynb` | Daily aggregates | 24 | 3/7/14d | LSTM Uni (3d), XGBoost (7d), LSTM Bi (14d) |
| `02_swarm_morning_window.ipynb` | 15-min + morning win. | 31 | 3d | LSTM Bi: G-Mean = 0.807, 13/14 events |
| `03_swarm_night_enhanced.ipynb` | 15-min + morn.+night | 44 | 3d | LSTM Uni: G-Mean = 0.823, 12/14 events |
| `04_honey_super.ipynb` | Daily aggregates | 19–52 | 7/14d | LightGBM (7d), XGBoost (14d) |

Each model notebook follows the same structure: data loading → feature engineering → **Model A** (XGBoost, 3-day horizon, baseline vs `weight_std_roll14`) → **Model A-LSTM** (Uni/Bi) → **Model B** (same-day anomaly detection) → walk-forward validation → event-level comparison → Optuna search.

---

## Behavioural Findings

### Weight–temperature correlation breakdown

Under normal foraging conditions the rolling 14-day Pearson correlation between daily hive weight and external temperature is positive (r ≈ +0.32): warmer days drive more nectar collection. In the 14 days before a swarm the sign reverses (r ≈ −0.34, p < 0.001): the colony stops responding to ambient temperature as it redirects energy toward queen-cell development and scout activity. After departure the correlation partially recovers (r ≈ +0.19). The same reversal was observed across all hives with multiple confirmed events (largest drop: Box3 Δz ≈ −0.86, Box13 Δz ≈ −0.49). This pattern is encoded in the `corr_w_temp` rolling-correlation feature.

![Weight–temperature scatter Box13: normal vs pre-swarm vs post-swarm](docs/images/correlation_breakdown.png)

### Nocturnal pre-swarm signals

Ferrari et al. (2008) report +150 Hz and +2 °C nocturnal changes before swarming. UAB data shows consistent internal temperature increases of +2–+8 °C across all events but hive-dependent frequency changes — none reaching the +150 Hz threshold. Box4 (+76 Hz) and Box3 (+59 Hz) show the largest frequency increases, while Box1 shows a decrease. This suggests the nocturnal temperature signal is more universal than the acoustic one, and that frequency amplitude depends on sensor placement and population size. Both findings motivated the nocturnal feature block in NB03.

![Nocturnal pre-swarm comparison vs Ferrari et al. (2008)](docs/images/literature_comparison.png)

---

## Methods

### Feature engineering

All variants share the same preprocessing pipeline: 15-minute grid, `ffill(limit=8)`, NaN for longer gaps. Features with >50% missing across the training split are dropped entirely.

**NB01 (24 daily features):** weight dynamics (moving averages 7/14d, differences 7/14/21d, rolling slope and acceleration, 3-day drop); acoustic signals (frequency and volume with 7d MA and trend); `corr_w_temp` (14-day rolling Pearson correlation between daily weight and external temperature); swarm history (`days_since_last_swarm`, `n_swarms_this_season`); external and internal temperature with 7-day trend. `days_since_last_swarm` uses sentinel 999 + binary flag `has_prior_swarm` to distinguish no history from long ago. For 7d and 14d horizons, positive training examples are augmented ±1 day; the 3d horizon is excluded from augmentation (signal too noisy at short range).

**NB02 (31 features):** NB01 + morning-window aggregations (10–14h): `win_Weight_drop`, `win_Weight_std`, `win_Weight_range`, `win_Freq_mean/max/std`, `win_Vol_mean/max`, and `freq_spike_ratio` (morning frequency max divided by its 7-day rolling median — detects anomalous acoustic bursting independent of per-hive baseline). Also adds `weight_std_roll14`, the only ablation candidate that consistently improves both XGBoost and LSTM; narrowing to 12–14h was rejected (higher false alarm rates).

**NB03 (44 features):** NB02 + 13 nocturnal aggregations (00–05h): mean and max nocturnal frequency with 3/7d MA, nocturnal frequency spike ratio, mean nocturnal volume, overnight weight and overnight delta with 3d smoothing, morning-to-night frequency ratio.

**NB04 (19–52 features after ablation):** daily aggregates for 17 hives, months 2–6 only. Four information sources: weight dynamics (1/7/14/21d differences, MA, rolling variability, trend slope/acceleration, positive-day streak); hive-relative historical features (`weight_vs_historical`, `weight_pct_of_max`); apiary-relative features (z-scores and percentile ranks against daily apiary mean — needed because honey super readiness is a comparative judgement); intervention timing (`days_since_last_alza`, `n_alzas_this_season`). Three binary flags from Gerardo's expert rules (`above_min_weight`, `overdue_for_alza`, `small_but_growing`). Temporal features (month, day-of-year, cyclical encodings) excluded via ablation — they caused the model to exploit calendar patterns rather than biological signals.

### Models

**XGBoost (swarm, NB01–03):** gradient-boosted trees with `scale_pos_weight` = negative/positive ratio of training split. NaN imputed with per-feature median on training split only. Optuna: 50 trials, TPE, objective = walk-forward Average Precision across 2025 and 2026 folds.

**LSTM (swarm, NB01–03):** unidirectional and bidirectional variants, 2 layers, hidden = 64, dropout = 0.3. NB01 uses sequences of 21 days; NB02/03 use 14 days. Features standardised with `StandardScaler` fitted on training split; residual NaN filled with 0. Optimiser: Adam, lr = 1e-3, weight decay = 1e-4. Class imbalance: weighted `BCEWithLogitsLoss`. Early stopping: 15 epochs without improvement on validation loss. Optuna (20 trials, TPE) searches hidden size, layers, dropout, lr, weight decay, and uni/bi direction. For NB02/03 the Optuna-tuned LSTM underperforms the fixed configuration on 2026 test — internal validation window too short to avoid search overfitting.

**XGBoost/LightGBM (honey super, NB04):** 6 feature configurations × 2 algorithms × 2 horizons (7d/14d) = 24 models. Selection criterion: G-Mean primary, then detection rate, then false alarm count.

### Evaluation

**Walk-forward cross-validation:** train on all data before year Y, test on year Y. Two folds for swarm models (test 2025, test 2026); primary results on 2026 (14 events, 3.92% positive). Honey super uses TimeSeriesSplit with 4 folds and a 14-day gap.

**Metrics:** AUC-ROC (discrimination at all thresholds); Average Precision (area under precision–recall curve, primary Optuna target); G-Mean = √(Sensitivity × Specificity), used for threshold selection and model ranking; event-level detection (swarms with at least one positive prediction in the prior 3-day window) paired with false alarm count (positives outside any event window).

**Threshold:** G-Mean-maximising threshold on walk-forward training folds, not the default 0.5. Reflects asymmetric costs — a missed swarm is a permanent productive loss; a false alarm is one unnecessary inspection.

---

## Results

### Multi-horizon swarm prediction (NB01)

All models evaluated on 2026 test set (hives with prior swarm history, seasonal window Feb–Jun, 14 events).

| Horizon | Model | AP | G-Mean | Det. | FA |
|---|---|---|---|---|---|
| 3d | XGBoost | 0.116 | 0.737 | 12/14 | 213 |
| 3d | **LSTM Uni** | **0.144** | **0.866** | **13/14** | **136** |
| 3d | LSTM Bi | 0.110 | 0.654 | 7/14 | 138 |
| 7d | **XGBoost** | **0.335** | **0.802** | **13/14** | 182 |
| 7d | LSTM Uni | 0.306 | 0.728 | 11/14 | 85 |
| 7d | LSTM Bi | 0.303 | 0.766 | 12/14 | **84** |
| 14d | XGBoost | 0.639 | 0.854 | **14/14** | 119 |
| 14d | LSTM Uni | 0.561 | 0.811 | 13/14 | 94 |
| 14d | **LSTM Bi** | 0.555 | 0.815 | 13/14 | **55** |

At 3d, LSTM Uni has the best G-Mean and event detection. At 7d, XGBoost leads in G-Mean and events despite lower AUC than LSTM (better-calibrated threshold). At 14d, LSTM Bi is selected for its false alarm count (55 vs 119) at near-equivalent AUC.

### Morning-window and night-enhanced models (NB02/03)

| NB | Model | AP | G-Mean | Det. | FA |
|---|---|---|---|---|---|
| 02 | XGBoost baseline | 0.112 | 0.616 | 11/14 | 437 |
| 02 | XGBoost + `weight_std_roll14` | 0.122 | 0.678 | 10/14 | **244** |
| 02 | LSTM Uni | 0.132 | 0.776 | **13/14** | 305 |
| 02 | **LSTM Bi** | **0.141** | **0.807** | **13/14** | 289 |
| 03 | XGBoost baseline | 0.134 | 0.647 | 10/14 | 284 |
| 03 | XGBoost + `weight_std_roll14` | 0.138 | 0.728 | 10/14 | 193 |
| 03 | **LSTM Uni** | **0.153** | **0.823** | 12/14 | **161** |
| 03 | LSTM Bi | 0.086 | 0.773 | **13/14** | 332 |

Adding `weight_std_roll14` improves XGBoost G-Mean by +0.024–0.045 and substantially reduces false alarms in both notebooks. LSTM consistently outperforms XGBoost. NB03 LSTM Uni achieves the highest AUC overall (0.882) but detects one fewer event than NB02 LSTM Bi; it reduces false alarms by 44% (161 vs 289).

### The simplicity result

Despite 44 features and nocturnal aggregations, NB03 does not surpass NB01 on the operational metrics. NB01 LSTM Uni (24 daily features, G-Mean = 0.866, 13/14 events, 136 FA) remains the best deployed model. Feature importance in NB03 XGBoost confirms this: nocturnal features do not appear until rank 6, while classical weight and temperature aggregates occupy the top five positions. The dominant pre-swarm signal is already captured in daily aggregations; higher temporal resolution adds noise rather than discriminatory power at the 3-day horizon.

### Honey super placement (NB04)

24 models evaluated on 20 real-world interventions from 2026.

**7-day window:**

| Model | Features | n | AP | G-Mean | Det. | FA |
|---|---|---|---|---|---|---|
| **LightGBM** | bio\_no\_tmp | 46 | **0.147** | **0.730** | 14/20 | 597 |
| XGBoost | bio\_no\_tmp | 46 | 0.108 | 0.729 | 14/20 | 625 |
| LightGBM | hive\_plus\_28 | 28 | 0.110 | 0.686 | 14/20 | 724 |
| XGBoost | hive\_19 | 19 | 0.106 | 0.657 | 11/20 | **390** |

**14-day window:**

| Model | Features | n | AP | G-Mean | Det. | FA |
|---|---|---|---|---|---|---|
| XGBoost | hive\_19 | 19 | 0.235 | **0.729** | 15/20 | 565 |
| LightGBM | hive\_19 | 19 | 0.243 | 0.728 | 13/20 | 395 |
| **XGBoost** | acoustic\_26 | 26 | **0.260** | 0.718 | **15/20** | **377** |
| LightGBM | acoustic\_26 | 26 | 0.260 | 0.700 | 13/20 | 432 |

4-fold walk-forward confirms moderate robustness (AUC ± std < 0.15 in both best models). Honey super prediction is inherently harder than swarm detection: unlike swarming, which is a biological event, honey super placement is a beekeeper management decision, so the model is partly learning human decision-making rather than a purely biological process.

### Best model per notebook (summary)

†\,=\,deployed to dashboard.

| Model | AUC | AP | G-Mean | Det. | FA |
|---|---|---|---|---|---|
| NB01 (3d) — LSTM Uni† | 0.876 | 0.144 | **0.866** | 13/14 | 136 |
| NB01 (7d) — XGBoost | 0.855 | 0.335 | **0.802** | 13/14 | 182 |
| NB01 (14d) — LSTM Bi† | 0.911 | 0.555 | 0.815 | 13/14 | **55** |
| NB02 — LSTM Bi | 0.859 | 0.141 | 0.807 | 13/14 | 289 |
| NB03 — LSTM Uni | 0.882 | **0.153** | 0.823 | 12/14 | 161 |
| NB04 (7d) — LightGBM† | 0.765 | 0.147 | 0.730 | 14/20 | 597 |
| NB04 (14d) — XGBoost† | 0.701 | 0.260 | 0.718 | **15/20** | 377 |

---

## Deployed System (API + Dashboard)

### Architecture

```
n8n (Monday 9am)
  ├── POST /weekly-run          → downloads 60d xlsx for 8 swarm hives via Playwright
  │                               → runs LSTM Uni (3d) + LSTM Bi (14d) per hive
  │                               → saves latest_results.json, emails alert if any ALTO risk at 3d
  └── POST /weekly-alza-update  → downloads 30d xlsx for 17 alza hives
                                  → rebuilds daily_features_prod.csv (~7 min)

FastAPI (port 8000)             Streamlit dashboard (port 8501)
  ├── /predict                    Tab 1 — Swarm Risk (3d + 14d probabilities per hive)
  ├── /weekly-run                 Tab 2 — Honey Supers (7d/14d recommendations)
  ├── /weekly-alza-update         Tab 3 — Sensor Data
  ├── /results                    Tab 4 — Risk Trends
  └── /history                    Tab 5 — Models (CV table + feature importance)
                                  Tab 6 — History (weekly run log)
```

### Running locally

```bash
# 1. Start the API
cd api
uvicorn main:app --host 0.0.0.0 --port 8000

# 2. Start the dashboard (separate terminal)
streamlit run dashboard.py --server.port 8501

# 3. Import n8n_workflow.json into a local n8n instance for weekly automation
#    (n8n start → http://localhost:5678)
```

Health check: `http://localhost:8000/health` → `{"status": "ok"}`

### Model artefacts in `api/models/`

| File | Description |
|---|---|
| `lstm_uni_01_3d.pt` | Swarm LSTM Uni — 3d horizon (NB01, G-Mean = 0.866, 13/14) |
| `lstm_bi_01_14d.pt` | Swarm LSTM Bi — 14d horizon (NB01, G-Mean = 0.815, 13/14) |
| `scaler_01.pkl` | StandardScaler fitted on NB01 training split |
| `median_fill_01.json` | Training-split medians for NaN imputation |
| `model_meta_01_3d.json` | Architecture params for the 3d model |
| `model_meta_01_14d.json` | Architecture params for the 14d model |
| `lgbm_alza_solo_hive_7d.pkl` | Honey super model — 7d horizon |
| `lgbm_alza_solo_hive_14d.pkl` | Honey super model — 14d horizon |
| `lgbm_alza_meta.json` | Alza model metadata (feature lists, thresholds) |

To retrain honey super models with updated data:

```bash
cd api
python train_alza_features.py   # grid search, saves best 7d + 14d to api/models/
```

---

## Requirements

```bash
pip install -r requirements.txt
```

**Key dependencies:** `pandas`, `numpy`, `xgboost`, `lightgbm`, `optuna`, `torch`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `joblib`

**API additional:** `fastapi`, `uvicorn`, `streamlit`, `playwright` (+ `python -m playwright install chromium`)

---

## Swarming Events Dataset

| Hive | Swarms | Period |
|---|---|---|
| Box1 | 2 | Apr 2026 |
| Box2 | 1 | May 2024 |
| Box3 | 3 | Apr 2025 |
| Box4 | 3 | Apr 2025 |
| Box5 | 2 | Apr–May 2026 |
| Box8 | 4 | Apr–May 2025–26 |
| Box13 | 4 | Apr 2025 |
| Box14 | 4 | Apr–May 2025–26 |
| **Total** | **23** | 2023–2026 |

Train/test split at 2026-01-01: training set = 3,308 rows, 9 events (0.76% positive); test set = 944 rows, 14 events (3.92% positive). Box1 and Box5 had no prior swarm history in the training data, making them the hardest cases for the model.

---

## Citation

```
Arpa Robig, M. (2026). Multi-Sensor Analysis and Early Detection of Swarming Events
in Honey Bee Colonies Using Machine Learning. Final Degree Project in Artificial
Intelligence, Universitat Autònoma de Barcelona.
```
