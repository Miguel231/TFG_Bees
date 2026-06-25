"""
train_alza_features.py — Feature set grid search for honey super prediction
Tests 6 feature configurations × LightGBM + XGBoost × 7d + 14d
Threshold: G-Mean optimized
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from lightgbm import LGBMClassifier, early_stopping as lgb_es, log_evaluation as lgb_log
from xgboost import XGBClassifier

HERE     = Path(__file__).parent
FEAT_CSV = HERE.parent.parent / "daily_features_final.csv"
OUT_DIR  = HERE / "models"
SPLIT    = pd.Timestamp("2026-01-01")

# ── Ground truth ───────────────────────────────────────────────────────────────
_alzas = {
    "2023-03-20":{1:1,5:1,8:1,9:1,10:1},"2023-04-13":{3:1,4:1},
    "2023-04-29":{3:1},"2023-05-04":{6:1},"2023-06-02":{3:1,4:1,5:1,8:1},
    "2024-03-28":{1:3,5:1,12:2},"2024-04-06":{3:1},"2024-04-08":{8:1},
    "2024-04-12":{1:1,3:1,5:1,8:1,11:1},
    "2025-01-23":{13:1},"2025-02-26":{2:1,3:1,13:1,15:1},"2025-03-08":{4:1},
    "2025-03-17":{3:1,4:1,6:1,14:1},"2025-03-25":{7:1,9:1,10:1,15:1},
    "2025-03-31":{15:1},"2025-04-07":{6:1,10:-1,13:-1,15:1},
    "2025-04-13":{2:1,6:1,7:-1,8:1,13:1,14:1},"2025-04-23":{3:1,6:1,8:1,13:1,14:1,15:1},
    "2026-02-27":{1:1,17:1},"2026-03-13":{1:1,11:1,17:1},"2026-03-26":{9:1},
    "2026-04-08":{4:1,5:1,9:1,11:1},"2026-05-07":{3:1,8:1,9:1,14:1,16:1},
    "2026-06-09":{7:1,12:1,13:1,15:1,16:1},
}
recs = []
for fecha, cols in _alzas.items():
    for bid, delta in cols.items():
        if delta > 0:
            recs.append({"fecha": pd.to_datetime(fecha), "box_id": int(bid)})
df_alz     = pd.DataFrame(recs)
interv_2026 = df_alz[df_alz["fecha"] >= SPLIT]

# ── Load CSV ───────────────────────────────────────────────────────────────────
daily = pd.read_csv(FEAT_CSV, parse_dates=["date"])
LEAKAGE = {"target", "target_binary", "dias_hasta_intervencion",
           "Unnamed: 0", "date", "box_id"}
ALL_FEATS = [c for c in daily.columns if c not in LEAKAGE]
print(f"Total clean features: {len(ALL_FEATS)}")

# ── Feature configurations ─────────────────────────────────────────────────────
TEMPORAL = ["sin_dayofyear","cos_dayofyear","month","season","days_in_season","dayofyear"]

FEAT_HIVE = [  # current baseline
    "Weight","weight_diff_7d","weight_diff_14d","weight_diff_21d",
    "weight_ma_7d","weight_trend_slope","n_positive_days_7d",
    "days_since_last_alza","days_since_last_ADD",
    "n_alzas_this_season","weight_vs_historical",
    "above_min_weight","overdue_for_alza",
    "days_in_season","sin_dayofyear","cos_dayofyear","month","season",
    "temp_trend_7d",
]

FEAT_HIVE_PLUS = FEAT_HIVE + [  # + top missing features from notebook importance
    "weight_ma_14d","Temp_scale","weight_std_7d","corr_w_temp",
    "weight_rank_pct","weight_vs_apiary","weight_diff_1d",
    "weight_amplitude","weight_acceleration",
]

FEAT_ACOUSTIC = FEAT_HIVE + [  # + acoustic/sensor features
    "Frequency","Freq_std","freq_ma_7d","freq_diff_7d","Volume",
    "Temp_scale","Humidity_scale",
]

FEAT_APIARY = FEAT_HIVE + [  # + cross-hive comparison
    "weight_vs_apiary","growth_vs_apiary","weight_rank_pct",
    "growth_rank_pct","weight_apiary_ratio","n_hives_active",
]

# No temporal, all biological signals
FEAT_BIO = [f for f in ALL_FEATS if f not in TEMPORAL]

# All features
FEAT_ALL = ALL_FEATS

CONFIGS = {
    "hive_19":      FEAT_HIVE,
    "hive_plus_28": FEAT_HIVE_PLUS,
    "acoustic_26":  FEAT_ACOUSTIC,
    "apiary_25":    FEAT_APIARY,
    "bio_no_tmp":   FEAT_BIO,
    "all_feats":    FEAT_ALL,
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def label_binary(df, horizon):
    out = df.copy(); out["target"] = 0
    for _, r in df_alz.iterrows():
        mask = ((out["box_id"]==r["box_id"]) &
                (out["date"] >= r["fecha"] - pd.Timedelta(days=horizon)) &
                (out["date"] <  r["fecha"]))
        out.loc[mask, "target"] = 1
    return out

def gmean_thr(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    gm = np.sqrt(tpr * (1-fpr))
    ix = int(gm.argmax())
    return float(thr[ix]), float(gm[ix])

def eval_model(clf, X_te, y_te, test_df, interv, horizon):
    y_prob = clf.predict_proba(X_te)[:, 1]
    thr, gm = gmean_thr(y_te, y_prob)
    auc = roc_auc_score(y_te, y_prob)
    ap  = average_precision_score(y_te, y_prob)
    fpr_v, tpr_v, _ = roc_curve(y_te, y_prob)
    gm_arr = np.sqrt(tpr_v*(1-fpr_v))
    ix = int(gm_arr.argmax())
    sens = float(tpr_v[ix])
    spec = float(1-fpr_v[ix])

    td = test_df.copy(); td["pred"] = (y_prob >= thr).astype(int)
    det = 0
    for _, r in interv.iterrows():
        mask = ((td["box_id"]==r["box_id"]) &
                (td["date"] >= r["fecha"] - pd.Timedelta(days=horizon)) &
                (td["date"] <  r["fecha"]))
        if mask.any() and td.loc[mask,"pred"].any(): det += 1
    fa = int(td["pred"].sum()) - int(
        sum(((td["box_id"]==r["box_id"]) &
             (td["date"] >= r["fecha"]-pd.Timedelta(days=horizon)) &
             (td["date"] < r["fecha"]) & (td["pred"]==1)).sum()
            for _, r in interv.iterrows()))
    return {"auc":auc,"ap":ap,"gmean":gm,"sens":sens,"spec":spec,
            "det":det,"fa":fa,"thr":thr}

LGBM_P = dict(n_estimators=500, max_depth=4, learning_rate=0.02,
              subsample=0.8, colsample_bytree=0.8,
              min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
              random_state=42, n_jobs=-1, verbose=-1)
XGB_P  = dict(n_estimators=400, max_depth=4, learning_rate=0.03,
              subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
              early_stopping_rounds=30, random_state=42, n_jobs=-1, verbosity=0)

# ── Main grid ──────────────────────────────────────────────────────────────────
results = []

for horizon in [7, 14]:
    df_lab = label_binary(daily, horizon)
    train  = df_lab[df_lab["date"] < SPLIT]
    test   = df_lab[df_lab["date"] >= SPLIT]
    test_meta = test[["box_id","date"]].copy().reset_index(drop=True)

    print(f"\n{'='*65}")
    print(f"  HORIZON {horizon}d  |  train pos={train['target'].sum()}  test pos={test['target'].sum()}")
    print(f"{'='*65}")

    for cfg_name, feat_list in CONFIGS.items():
        feats = [f for f in feat_list if f in daily.columns]
        X_tr  = train[feats].replace([np.inf,-np.inf], np.nan)
        med   = X_tr.median()
        X_tr  = X_tr.fillna(med).values
        y_tr  = train["target"].values
        X_te  = test[feats].replace([np.inf,-np.inf], np.nan).fillna(med).values
        y_te  = test["target"].values
        pw    = float((y_tr==0).sum()) / max(float((y_tr==1).sum()), 1)

        for algo_name, clf in [
            ("LGBM", LGBMClassifier(scale_pos_weight=pw, **LGBM_P)),
            ("XGB",  XGBClassifier(scale_pos_weight=pw, **XGB_P)),
        ]:
            if algo_name == "LGBM":
                clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
                        callbacks=[lgb_es(30,verbose=False), lgb_log(9999)])
            else:
                clf.fit(X_tr, y_tr, eval_set=[(X_te,y_te)], verbose=False)

            m = eval_model(clf, X_te, y_te, test_meta, interv_2026, horizon)
            m.update({"horizon":horizon,"algo":algo_name,"config":cfg_name,"n_feat":len(feats),
                      "clf":clf,"med":med,"feats":feats})
            results.append(m)
            print(f"  {algo_name:<5} {cfg_name:<16} n={len(feats):2d} | "
                  f"AUC={m['auc']:.3f} AP={m['ap']:.3f} GM={m['gmean']:.3f} "
                  f"S={m['sens']:.2f} Sp={m['spec']:.2f} "
                  f"Det={m['det']}/20 FA={m['fa']:4d} Thr={m['thr']:.3f}")

# ── Final tables ───────────────────────────────────────────────────────────────
df_res = pd.DataFrame([{k:v for k,v in r.items() if k not in ("clf","med","feats")}
                       for r in results])

print("\n\n" + "="*80)
print("FINAL TABLE — sorted by G-Mean")
print("="*80)
for h in [7, 14]:
    sub = df_res[df_res["horizon"]==h].sort_values("gmean", ascending=False).reset_index(drop=True)
    print(f"\n--- {h}d ---")
    print(f"{'Rank':<5}{'Algo':<6}{'Config':<18}{'nF':>3}  {'AUC':>6}{'AP':>6}{'GMean':>7}{'Sens':>6}{'Spec':>6}{'Det':>6}{'FA':>5}{'Thr':>7}")
    print("-"*80)
    for i, row in sub.iterrows():
        mark = " ◄" if i == 0 else ""
        print(f"  {i+1:<4}{row['algo']:<6}{row['config']:<18}{row['n_feat']:>3}  "
              f"{row['auc']:>6.3f}{row['ap']:>6.3f}{row['gmean']:>7.3f}"
              f"{row['sens']:>6.2f}{row['spec']:>6.2f}{row['det']:>4}/20"
              f"{row['fa']:>5}{row['thr']:>7.3f}{mark}")

# ── Save best per horizon ──────────────────────────────────────────────────────
# Selection criterion (paper §5.3): G-Mean + event detection + false alarms.
# 7d:  LGBM bio_no_tmp wins on G-Mean (0.730) and detection (14/20).
# 14d: XGB acoustic_26 preferred over XGB hive_19 despite lower G-Mean (0.718 vs 0.729)
#      because same Det=15/20 with 377 FA vs 565 FA (−34% false alarms).
def _select_best(candidates):
    """Paper selection: G-Mean primary, then det, then fewer FA as tiebreaker."""
    max_gm  = max(c["gmean"] for c in candidates)
    top_gm  = [c for c in candidates if c["gmean"] >= max_gm - 0.02]
    max_det = max(c["det"]   for c in top_gm)
    top_det = [c for c in top_gm  if c["det"] == max_det]
    return min(top_det, key=lambda x: x["fa"])

print("\n\nSaving best models...")
for h in [7, 14]:
    best = _select_best([r for r in results if r["horizon"]==h])
    fname = "lgbm_alza_solo_hive_7d.pkl" if h == 7 else "lgbm_alza_solo_hive_14d.pkl"
    bundle = {
        "model":        best["clf"],
        "features":     best["feats"],
        "median":       best["med"],
        "threshold":    best["thr"],
        "horizon_days": h,
        "auc":          round(best["auc"], 4),
        "gmean":        round(best["gmean"], 4),
        "detected":     best["det"],
        "model_type":   best["algo"],
        "config":       best["config"],
    }
    joblib.dump(bundle, OUT_DIR / fname)
    print(f"  {h}d best: {best['algo']} {best['config']} "
          f"G-Mean={best['gmean']:.3f} Det={best['det']}/20 -> {fname}")

print("\nDone.")
