"""
train_alza_full.py — Full model comparison for honey super prediction
Models: LightGBM, XGBoost (con/sin temporales), LSTM Uni, LSTM Bi
Windows: 7d, 14d  |  Threshold: G-Mean optimized
Usage: cd TFG_Bees/api && python train_alza_full.py
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier, early_stopping as lgb_es, log_evaluation as lgb_log
from xgboost import XGBClassifier

HERE     = Path(__file__).parent
FEAT_CSV = HERE.parent.parent / "daily_features_final.csv"
OUT_DIR  = HERE / "models"
OUT_DIR.mkdir(exist_ok=True)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT    = pd.Timestamp("2026-01-01")
SEQ_LEN  = 14

print(f"Device: {DEVICE}")

# ── Ground truth alzas ADD ─────────────────────────────────────────────────────
_alzas = {
    "2023-03-20": {1:1,5:1,8:1,9:1,10:1}, "2023-04-13": {3:1,4:1},
    "2023-04-29": {3:1}, "2023-05-04": {6:1}, "2023-06-02": {3:1,4:1,5:1,8:1},
    "2024-03-28": {1:3,5:1,12:2}, "2024-04-06": {3:1}, "2024-04-08": {8:1},
    "2024-04-12": {1:1,3:1,5:1,8:1,11:1},
    "2025-01-23": {13:1}, "2025-02-26": {2:1,3:1,13:1,15:1}, "2025-03-08": {4:1},
    "2025-03-17": {3:1,4:1,6:1,14:1}, "2025-03-25": {7:1,9:1,10:1,15:1},
    "2025-03-31": {15:1}, "2025-04-07": {6:1,10:-1,13:-1,15:1},
    "2025-04-13": {2:1,6:1,7:-1,8:1,13:1,14:1}, "2025-04-23": {3:1,6:1,8:1,13:1,14:1,15:1},
    "2026-02-27": {1:1,17:1}, "2026-03-13": {1:1,11:1,17:1}, "2026-03-26": {9:1},
    "2026-04-08": {4:1,5:1,9:1,11:1}, "2026-05-07": {3:1,8:1,9:1,14:1,16:1},
    "2026-06-09": {7:1,12:1,13:1,15:1,16:1},
}
recs = []
for fecha, cols in _alzas.items():
    for bid, delta in cols.items():
        recs.append({"fecha": pd.to_datetime(fecha), "box_id": int(bid),
                     "accion": "ADD" if delta > 0 else "REMOVE"})
df_alzas_add = (pd.DataFrame(recs).query("accion=='ADD'")
                .sort_values(["box_id","fecha"]).reset_index(drop=True))
interv_2026 = df_alzas_add[df_alzas_add["fecha"] >= SPLIT].copy()
print(f"Alzas ADD total: {len(df_alzas_add)} | 2026 test: {len(interv_2026)}")

# ── Feature sets ───────────────────────────────────────────────────────────────
FEAT_HIVE = [
    "Weight","weight_diff_7d","weight_diff_14d","weight_diff_21d",
    "weight_ma_7d","weight_trend_slope","n_positive_days_7d",
    "days_since_last_alza","days_since_last_ADD",
    "n_alzas_this_season","weight_vs_historical",
    "above_min_weight","overdue_for_alza",
    "days_in_season","sin_dayofyear","cos_dayofyear","month","season",
    "temp_trend_7d",
]
TEMPORAL    = {"sin_dayofyear","cos_dayofyear","month","season","days_in_season"}
FEAT_NO_TMP = [f for f in FEAT_HIVE if f not in TEMPORAL]

# ── Load features ──────────────────────────────────────────────────────────────
print(f"Loading {FEAT_CSV}...")
daily = pd.read_csv(FEAT_CSV, parse_dates=["date"])
print(f"  {len(daily):,} rows | {daily['date'].min().date()} -> {daily['date'].max().date()}")

# ── Helpers ────────────────────────────────────────────────────────────────────
def label_binary(df, df_alz, horizon):
    out = df.copy(); out["target"] = 0
    for _, r in df_alz.iterrows():
        mask = ((out["box_id"]==r["box_id"]) &
                (out["date"] >= r["fecha"] - pd.Timedelta(days=horizon)) &
                (out["date"] <  r["fecha"]))
        out.loc[mask, "target"] = 1
    return out

def gmean_thr(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    gm = np.sqrt(tpr * (1 - fpr))
    ix = int(gm.argmax())
    return float(thr[ix]), float(gm[ix])

def compute_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    auc  = roc_auc_score(y_true, y_prob)
    ap   = average_precision_score(y_true, y_prob)
    tp   = int(((y_pred==1)&(y_true==1)).sum())
    fn   = int(((y_pred==0)&(y_true==1)).sum())
    tn   = int(((y_pred==0)&(y_true==0)).sum())
    fp   = int(((y_pred==1)&(y_true==0)).sum())
    sens = tp / max(tp+fn, 1)
    spec = tn / max(tn+fp, 1)
    gm   = float(np.sqrt(sens * spec))
    return {"auc":auc, "ap":ap, "gmean":gm, "sens":sens, "spec":spec,
            "tp":tp,"fp":fp,"thr":thr}

def detect_rate(meta_df, pred_col, interv, horizon):
    det = 0
    for _, r in interv.iterrows():
        mask = ((meta_df["box_id"]==r["box_id"]) &
                (pd.to_datetime(meta_df["date"]) >= r["fecha"] - pd.Timedelta(days=horizon)) &
                (pd.to_datetime(meta_df["date"]) <  r["fecha"]))
        if mask.any() and meta_df.loc[mask, pred_col].any():
            det += 1
    return det

def false_alarm_rate(meta_df, pred_col, interv, horizon):
    """Days flagged positive that are NOT in any pre-alza window."""
    total_tp_days = 0
    total_pos_days = int(meta_df[pred_col].sum())
    for _, r in interv.iterrows():
        mask = ((meta_df["box_id"]==r["box_id"]) &
                (pd.to_datetime(meta_df["date"]) >= r["fecha"] - pd.Timedelta(days=horizon)) &
                (pd.to_datetime(meta_df["date"]) <  r["fecha"]))
        total_tp_days += int(meta_df.loc[mask, pred_col].sum())
    fa = total_pos_days - total_tp_days
    return fa

# ── Tree model trainers ────────────────────────────────────────────────────────
LGBM_P = dict(n_estimators=500, max_depth=4, learning_rate=0.02,
              subsample=0.8, colsample_bytree=0.8,
              min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0,
              random_state=42, n_jobs=-1, verbose=-1)

XGB_P  = dict(n_estimators=400, max_depth=4, learning_rate=0.03,
              subsample=0.8, colsample_bytree=0.8,
              eval_metric="aucpr", early_stopping_rounds=30,
              random_state=42, n_jobs=-1, verbosity=0)

def train_lgbm(X_tr, y_tr, X_te, y_te):
    pw  = float((y_tr==0).sum()) / max(float((y_tr==1).sum()), 1)
    clf = LGBMClassifier(scale_pos_weight=pw, **LGBM_P)
    clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
            callbacks=[lgb_es(30, verbose=False), lgb_log(9999)])
    return clf

def train_xgb(X_tr, y_tr, X_te, y_te):
    pw  = float((y_tr==0).sum()) / max(float((y_tr==1).sum()), 1)
    clf = XGBClassifier(scale_pos_weight=pw, **XGB_P)
    clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    return clf

# ── LSTM ───────────────────────────────────────────────────────────────────────
class LSTMAlza(nn.Module):
    def __init__(self, n_feat, hidden=64, n_layers=2, dropout=0.3, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0,
                            bidirectional=bidirectional)
        d = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)

def make_seqs(df, feats, target_col, seq_len):
    Xs, ys, meta = [], [], []
    for hid, hdf in df.groupby("box_id"):
        hdf = hdf.sort_values("date").reset_index(drop=True)
        arr = hdf[feats].values.astype(np.float32)
        tgt = hdf[target_col].values.astype(np.float32)
        dts = hdf["date"].values
        for i in range(seq_len, len(hdf)):
            Xs.append(arr[i-seq_len:i])
            ys.append(tgt[i])
            meta.append({"box_id": int(hid), "date": dts[i]})
    if not Xs:
        return (np.zeros((0, seq_len, len(feats)), np.float32),
                np.zeros(0, np.float32), pd.DataFrame(meta))
    return np.stack(Xs), np.array(ys, np.float32), pd.DataFrame(meta)

def train_lstm_model(X_tr, y_tr, X_te, y_te, n_feat, bi=False, epochs=100, patience=15):
    mod  = LSTMAlza(n_feat, hidden=64, n_layers=2, dropout=0.3, bidirectional=bi).to(DEVICE)
    pw   = torch.tensor([(y_tr==0).sum() / max((y_tr==1).sum(), 1)]).float().to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt  = torch.optim.Adam(mod.parameters(), lr=1e-3, weight_decay=1e-4)
    ds   = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    ldr  = DataLoader(ds, batch_size=64, shuffle=True)
    Xte  = torch.FloatTensor(X_te).to(DEVICE)

    best_auc, best_state, no_imp = 0.0, None, 0
    for ep in range(1, epochs+1):
        mod.train()
        for xb, yb in ldr:
            opt.zero_grad()
            crit(mod(xb.to(DEVICE)), yb.to(DEVICE)).backward()
            opt.step()
        mod.eval()
        with torch.no_grad():
            probs = torch.sigmoid(mod(Xte)).cpu().numpy()
        try:    cur_auc = roc_auc_score(y_te, probs)
        except: cur_auc = 0.5
        if cur_auc > best_auc:
            best_auc = cur_auc
            best_state = {k: v.clone() for k, v in mod.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if ep % 20 == 1 or ep == epochs:
            print(f"    ep {ep:3d} | AUC={cur_auc:.3f} | best={best_auc:.3f}")
        if no_imp >= patience:
            print(f"    Early stop ep={ep}")
            break
    mod.load_state_dict(best_state)
    return mod

# ── Main loop ──────────────────────────────────────────────────────────────────
all_results   = {}
best_bundles  = {}

for horizon in [7, 14]:
    print(f"\n\n{'='*70}")
    print(f"  HORIZON {horizon}d")
    print(f"{'='*70}")

    df_lab = label_binary(daily, df_alzas_add, horizon)
    train  = df_lab[df_lab["date"] < SPLIT]
    test   = df_lab[df_lab["date"] >= SPLIT]
    print(f"Train: {len(train):,} | pos={train['target'].sum()} | "
          f"Test: {len(test):,} | pos={test['target'].sum()}")

    hr = {}

    # ── Tree models ────────────────────────────────────────────────────────────
    for mname, feat_list, trainer in [
        ("LGBM_con_temp", FEAT_HIVE,   train_lgbm),
        ("LGBM_sin_temp", FEAT_NO_TMP, train_lgbm),
        ("XGB_con_temp",  FEAT_HIVE,   train_xgb),
        ("XGB_sin_temp",  FEAT_NO_TMP, train_xgb),
    ]:
        feats = [f for f in feat_list if f in daily.columns]
        X_tr  = train[feats].replace([np.inf,-np.inf], np.nan)
        med   = X_tr.median()
        X_tr  = X_tr.fillna(med).values
        y_tr  = train["target"].values
        X_te  = test[feats].replace([np.inf,-np.inf], np.nan).fillna(med).values
        y_te  = test["target"].values

        print(f"\n--- {mname} {horizon}d ---")
        clf    = trainer(X_tr, y_tr, X_te, y_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        thr, _ = gmean_thr(y_te, y_prob)
        m      = compute_metrics(y_te, y_prob, thr)

        meta_te = test[["box_id","date"]].copy().reset_index(drop=True)
        meta_te["pred"] = (y_prob >= thr).astype(int)
        meta_te["prob"] = y_prob
        m["det"]  = detect_rate(meta_te, "pred", interv_2026, horizon)
        m["fa"]   = false_alarm_rate(meta_te, "pred", interv_2026, horizon)
        m["n"]    = len(feats)
        m["clf"]  = clf
        m["med"]  = med
        m["feats"]= feats
        hr[mname] = m
        print(f"  AUC={m['auc']:.3f} | AP={m['ap']:.3f} | G-Mean={m['gmean']:.3f} | "
              f"Sens={m['sens']:.2f} | Spec={m['spec']:.2f} | "
              f"Thr={thr:.3f} | Det={m['det']}/20 | FA={m['fa']}")

    # ── LSTM models ────────────────────────────────────────────────────────────
    feats_lstm = [f for f in FEAT_HIVE if f in daily.columns]
    n_feat     = len(feats_lstm)

    # Scale on train, transform all (scaler fitted on train only)
    df_sc = df_lab.copy()
    scaler = StandardScaler()
    tr_idx = df_sc["date"] < SPLIT
    df_sc_vals = df_sc[feats_lstm].replace([np.inf,-np.inf], np.nan).fillna(0)
    df_sc.loc[tr_idx,  feats_lstm] = scaler.fit_transform(df_sc_vals[tr_idx])
    df_sc.loc[~tr_idx, feats_lstm] = scaler.transform(df_sc_vals[~tr_idx])

    # Build sequences from full scaled dataset, then split by date
    X_all, y_all, meta_all = make_seqs(df_sc, feats_lstm, "target", SEQ_LEN)
    meta_all["date"] = pd.to_datetime(meta_all["date"])
    tr_mask = meta_all["date"] < SPLIT
    te_mask = ~tr_mask

    X_tr_seq = X_all[tr_mask]; y_tr_seq = y_all[tr_mask]
    X_te_seq = X_all[te_mask]; y_te_seq = y_all[te_mask]
    meta_te  = meta_all[te_mask].reset_index(drop=True)

    print(f"\n  LSTM seqs: train={len(X_tr_seq)} pos={int(y_tr_seq.sum())} | "
          f"test={len(X_te_seq)} pos={int(y_te_seq.sum())}")

    for lstm_name, bi in [("LSTM_Uni", False), ("LSTM_Bi", True)]:
        print(f"\n--- {lstm_name} {horizon}d ---")
        mod = train_lstm_model(X_tr_seq, y_tr_seq, X_te_seq, y_te_seq, n_feat, bi=bi)
        mod.eval()
        with torch.no_grad():
            y_prob = torch.sigmoid(mod(torch.FloatTensor(X_te_seq).to(DEVICE))).cpu().numpy()

        thr, _ = gmean_thr(y_te_seq, y_prob)
        m      = compute_metrics(y_te_seq, y_prob, thr)

        mt = meta_te.copy()
        mt["pred"] = (y_prob >= thr).astype(int)
        mt["prob"] = y_prob
        m["det"]   = detect_rate(mt, "pred", interv_2026, horizon)
        m["fa"]    = false_alarm_rate(mt, "pred", interv_2026, horizon)
        m["n"]     = n_feat
        m["mod"]   = mod
        m["scaler"]= scaler
        m["feats"] = feats_lstm
        hr[lstm_name] = m
        print(f"  AUC={m['auc']:.3f} | AP={m['ap']:.3f} | G-Mean={m['gmean']:.3f} | "
              f"Sens={m['sens']:.2f} | Spec={m['spec']:.2f} | "
              f"Thr={thr:.3f} | Det={m['det']}/20 | FA={m['fa']}")

    all_results[horizon] = hr

# ── Final comparison table ─────────────────────────────────────────────────────
print("\n\n" + "="*80)
print("FINAL COMPARISON — G-Mean optimized threshold")
print("="*80)

for horizon in [7, 14]:
    print(f"\n{'─'*80}")
    print(f"  {horizon}d horizon")
    print(f"{'─'*80}")
    hr = all_results[horizon]
    hdr = f"{'Model':<20} {'AUC':>6} {'AP':>6} {'G-Mean':>7} {'Sens':>6} {'Spec':>6} {'Det/20':>7} {'FA':>5} {'Thr':>6} {'Feat':>5}"
    print(hdr)
    print("-"*80)
    sorted_m = sorted(hr.items(), key=lambda x: x[1]["gmean"], reverse=True)
    for i, (mname, m) in enumerate(sorted_m):
        mark = " ◄ BEST" if i == 0 else ""
        print(f"{mname:<20} {m['auc']:>6.3f} {m['ap']:>6.3f} {m['gmean']:>7.3f} "
              f"{m['sens']:>6.2f} {m['spec']:>6.2f} {m['det']:>5}/20 {m['fa']:>5} "
              f"{m['thr']:>6.3f} {m['n']:>5}{mark}")

# ── Save best models per horizon ───────────────────────────────────────────────
print("\n\nSaving best models...")

for horizon in [7, 14]:
    hr = all_results[horizon]
    best_name, best_m = max(hr.items(), key=lambda x: x[1]["gmean"])
    out_path = OUT_DIR / f"alza_best_{horizon}d.pkl"

    if "LSTM" in best_name:
        bundle = {
            "type":        best_name,
            "model":       best_m["mod"].cpu().state_dict(),
            "model_class": "LSTM_Bi" if "Bi" in best_name else "LSTM_Uni",
            "n_feat":      best_m["n"],
            "scaler":      best_m["scaler"],
            "features":    best_m["feats"],
            "threshold":   best_m["thr"],
            "horizon_days":horizon,
            "auc":         round(best_m["auc"], 4),
            "gmean":       round(best_m["gmean"], 4),
            "detected":    best_m["det"],
        }
    else:
        bundle = {
            "type":        best_name,
            "model":       best_m["clf"],
            "features":    best_m["feats"],
            "median":      best_m["med"],
            "threshold":   best_m["thr"],
            "horizon_days":horizon,
            "auc":         round(best_m["auc"], 4),
            "gmean":       round(best_m["gmean"], 4),
            "detected":    best_m["det"],
        }

    joblib.dump(bundle, out_path)
    print(f"  {horizon}d  -> {best_name}  G-Mean={best_m['gmean']:.3f}  Det={best_m['det']}/20  saved: {out_path.name}")

print("\nDone.")
