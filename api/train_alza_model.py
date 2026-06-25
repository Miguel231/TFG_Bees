"""
Train LightGBM honey-super (alza) model with 7d and 14d horizons.
Saves the better one to api/models/lgbm_alza_solo_hive_best.pkl
Usage: cd TFG_Bees/api && python train_alza_model.py
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stop, log_evaluation as lgb_log
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
FEAT_CSV  = HERE.parent.parent / "daily_features_final.csv"
OUT_DIR   = HERE / "models"
OUT_DIR.mkdir(exist_ok=True)

# ── Alzas ground truth (from notebook cells 5, 6, 8) ─────────────────────────
alzas_2023 = {
    "2023-03-20": {1:1, 5:1, 8:1, 9:1, 10:1},
    "2023-04-13": {3:1, 4:1},
    "2023-04-29": {3:1},
    "2023-05-04": {6:1},
    "2023-06-02": {3:1, 4:1, 5:1, 8:1},
}
alzas_2024 = {
    "2024-03-28": {1:3, 5:1, 12:2},
    "2024-04-06": {3:1},
    "2024-04-08": {8:1},
    "2024-04-12": {1:1, 3:1, 5:1, 8:1, 11:1},
}
alzas_2025 = {
    "2025-01-23": {13:1},
    "2025-02-26": {2:1, 3:1, 13:1, 15:1},
    "2025-03-08": {4:1},
    "2025-03-17": {3:1, 4:1, 6:1, 14:1},
    "2025-03-25": {7:1, 9:1, 10:1, 15:1},
    "2025-03-31": {15:1},
    "2025-04-07": {6:1, 10:-1, 13:-1, 15:1},
    "2025-04-13": {2:1, 6:1, 7:-1, 8:1, 13:1, 14:1},
    "2025-04-23": {3:1, 6:1, 8:1, 13:1, 14:1, 15:1},
}
alzas_2026 = {
    "2026-02-27": {1:1, 17:1},
    "2026-03-13": {1:1, 11:1, 17:1},
    "2026-03-26": {9:1},
    "2026-04-08": {4:1, 5:1, 9:1, 11:1},
    "2026-05-07": {3:1, 8:1, 9:1, 14:1, 16:1},
    "2026-06-09": {7:1, 12:1, 13:1, 15:1, 16:1},
}

alzas_todas = {**alzas_2023, **alzas_2024, **alzas_2025, **alzas_2026}
registros = []
for fecha, colmenas in alzas_todas.items():
    for box_id, delta in colmenas.items():
        registros.append({
            "fecha":   pd.to_datetime(fecha),
            "box_id":  int(box_id),
            "accion":  "ADD" if delta > 0 else "REMOVE",
        })
df_alzas = pd.DataFrame(registros).sort_values(["box_id", "fecha"]).reset_index(drop=True)
df_alzas_add = df_alzas[df_alzas["accion"] == "ADD"].copy()
print(f"Alzas ADD: {len(df_alzas_add)} | REMOVE: {len(df_alzas) - len(df_alzas_add)}")

# ── Features ───────────────────────────────────────────────────────────────────
FEATURES = [
    "Weight", "weight_diff_7d", "weight_diff_14d", "weight_diff_21d",
    "weight_ma_7d", "weight_trend_slope", "n_positive_days_7d",
    "days_since_last_alza", "days_since_last_ADD",
    "n_alzas_this_season", "weight_vs_historical",
    "above_min_weight", "overdue_for_alza",
    "days_in_season", "sin_dayofyear", "cos_dayofyear", "month", "season",
    "temp_trend_7d",
]

# ── Load pre-computed daily features ──────────────────────────────────────────
print(f"Loading {FEAT_CSV} ...")
daily = pd.read_csv(FEAT_CSV, parse_dates=["date"])
print(f"  {len(daily):,} rows, {daily['date'].min().date()} to {daily['date'].max().date()}")

feats_avail = [f for f in FEATURES if f in daily.columns]
print(f"  Features available: {len(feats_avail)}/{len(FEATURES)}")
missing = [f for f in FEATURES if f not in daily.columns]
if missing:
    print(f"  Missing (will skip): {missing}")

# ── Labelling helper ───────────────────────────────────────────────────────────
def label_binary(df, df_alzas_add, dias_pre):
    out = df.copy()
    out["target"] = 0
    for _, interv in df_alzas_add.iterrows():
        mask = (
            (out["box_id"] == interv["box_id"]) &
            (out["date"] >= interv["fecha"] - pd.Timedelta(days=dias_pre)) &
            (out["date"] <  interv["fecha"])
        )
        out.loc[mask, "target"] = 1
    return out

LGBM_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8,
    min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbose=-1,
)
SPLIT = "2026-01-01"

results = {}
models  = {}

for dias in [7, 14]:
    print(f"\n{'='*55}")
    print(f"  Horizon {dias}d")
    print(f"{'='*55}")

    df_lab = label_binary(daily, df_alzas_add, dias)
    train  = df_lab[df_lab["date"] < SPLIT]
    test   = df_lab[df_lab["date"] >= SPLIT]

    X_tr = train[feats_avail].replace([np.inf, -np.inf], np.nan)
    med  = X_tr.median()
    X_tr = X_tr.fillna(med)
    y_tr = train["target"]

    X_te = test[feats_avail].replace([np.inf, -np.inf], np.nan).fillna(med)
    y_te = test["target"]

    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    clf = LGBMClassifier(scale_pos_weight=pos_w, **LGBM_PARAMS)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        callbacks=[lgb_early_stop(30, verbose=False), lgb_log(9999)],
    )

    y_prob = clf.predict_proba(X_te)[:, 1]
    auc    = roc_auc_score(y_te, y_prob)
    prec, rec, thr = precision_recall_curve(y_te, y_prob)
    f1  = 2 * prec * rec / (prec + rec + 1e-8)
    bi  = f1.argmax()
    thr_best = thr[bi] if bi < len(thr) else 0.5
    y_pred   = (y_prob >= thr_best).astype(int)

    # Detection rate on 2026 interventions
    interv_2026 = df_alzas_add[df_alzas_add["fecha"] >= pd.to_datetime("2026-01-01")]
    test_pred = test.copy()
    test_pred["pred"] = y_pred
    detected = 0
    for _, row in interv_2026.iterrows():
        mask = (
            (test_pred["box_id"] == row["box_id"]) &
            (test_pred["date"] >= row["fecha"] - pd.Timedelta(days=dias)) &
            (test_pred["date"] <  row["fecha"])
        )
        if mask.any() and test_pred.loc[mask, "pred"].any():
            detected += 1

    print(f"  Train positives: {y_tr.sum()} / {len(y_tr)}")
    print(f"  Test  positives: {y_te.sum()} / {len(y_te)}")
    print(f"  AUC:      {auc:.3f}")
    print(f"  Recall:   {rec[bi]:.2f}  Precision: {prec[bi]:.2f}")
    print(f"  Detected: {detected}/{len(interv_2026)} 2026 interventions")
    print(f"  Threshold: {thr_best:.3f}")
    print()
    print(classification_report(y_te, y_pred, target_names=["NO ACTION", "ADD SUPER"], zero_division=0))

    results[dias] = {"auc": auc, "detected": detected, "threshold": thr_best, "median": med}
    models[dias]  = clf

# ── Save both models with fixed 50% threshold ─────────────────────────────────
import json
THR_FIXED = 0.50

for dias in [7, 14]:
    out_path = OUT_DIR / f"lgbm_alza_solo_hive_{dias}d.pkl"
    joblib.dump({
        "model":        models[dias],
        "features":     feats_avail,
        "threshold":    THR_FIXED,
        "median":       results[dias]["median"],
        "horizon_days": dias,
        "auc":          round(results[dias]["auc"], 4),
        "detected":     results[dias]["detected"],
    }, out_path)
    print(f"Saved {dias}d: {out_path}")

print(f"\nBoth models saved with threshold={THR_FIXED}")
print(f"  7d  AUC={results[7]['auc']:.3f}  detected={results[7]['detected']}/20")
print(f"  14d AUC={results[14]['auc']:.3f}  detected={results[14]['detected']}/20")
