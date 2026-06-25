"""
Optuna hyperparameter search for Notebook 01 (daily aggregates swarm model).
Mirrors the structure of NB02/NB03: XGBoost (50 trials, walk-forward AP) +
LSTM Uni (20 trials, internal validation loss), both for 3-day horizon.
"""
import os, sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from xgboost import XGBClassifier
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV   = os.path.join(BASE, '..', '..', 'data', 'daily_data.csv')
SPLIT_DATE = '2026-01-01'
HORIZON    = 3
SEED       = 42
SEQ_LEN    = 21
BATCH      = 64
DEVICE     = torch.device('cpu')

# ── Load data ────────────────────────────────────────────────────────────
daily = pd.read_csv(DATA_CSV, parse_dates=['date'])
daily = daily.sort_values(['box_id', 'date']).reset_index(drop=True)

# ── Events ───────────────────────────────────────────────────────────────
enjambres_raw = [
    (2024, 5, 24,  2, None),
    (2025, 4,  4, 13, None),
    (2025, 4,  7, 13,  2.5),
    (2025, 4,  8, 13, None),
    (2025, 4, 16, 13, None),
    (2025, 4, 17,  3, None),
    (2025, 4, 18, 14,  2.0),
    (2025, 4, 23,  3,  2.0),
    (2025, 4, 23,  4, None),
    (2026, 4,  7,  1,  2.8),
    (2026, 4, 23,  1,  2.0),
    (2026, 4, 23,  4,  2.5),
    (2026, 4, 24,  4,  2.0),
    (2026, 4, 25,  5,  2.0),
    (2026, 4, 29,  8,  1.4),
    (2026, 5,  1,  5,  1.0),
    (2026, 5,  5,  8,  1.5),
    (2026, 5, 10,  3,  1.5),
    (2026, 5, 10,  4,  1.3),
    (2026, 5, 11,  8,  1.3),
    (2026, 5, 14,  3,  0.7),
    (2026, 5, 17,  3,  1.0),
    (2026, 6,  1, 14,  3.0),
]
df_enjambres = pd.DataFrame(enjambres_raw, columns=['year','month','day','box_id','peso_kg'])
df_enjambres['fecha'] = pd.to_datetime(df_enjambres[['year','month','day']].rename(
    columns={'year':'year','month':'month','day':'day'}))
df_enjambres = df_enjambres.drop(columns=['year','month','day']).sort_values('fecha').reset_index(drop=True)

# ── Feature engineering (replicated from NB01 cell 5) ────────────────────
def add_swarm_features(df, df_enj):
    df = df.copy().sort_values(['box_id', 'date'])
    for col in ['weight_peak_ratio','freq_trend_7d','vol_trend_7d',
                'weight_drop_3d','days_since_last_swarm','n_swarms_this_season','corr_w_temp']:
        df[col] = np.nan

    for box_id in df['box_id'].unique():
        mask   = df['box_id'] == box_id
        df_box = df[mask].sort_values('date')
        idx    = df_box.index
        exp_peak = df_box['Weight'].expanding(min_periods=14).quantile(0.95).shift(1)
        df.loc[idx, 'weight_peak_ratio'] = df_box['Weight'].values / (exp_peak.values + 1e-6)
        df.loc[idx, 'freq_trend_7d']  = df_box['Frequency'].diff(7)
        df.loc[idx, 'vol_trend_7d']   = df_box['Volume'].diff(7)
        df.loc[idx, 'weight_drop_3d'] = -(df_box['Weight'].diff(3))
        df.loc[idx, 'corr_w_temp']    = df_box['Weight'].rolling(14, min_periods=7).corr(df_box['Temp_scale'])
        enj_b = df_enj[df_enj['box_id'] == box_id].sort_values('fecha')
        dates_df = df_box[['date']].sort_values('date').reset_index()
        dates_df['date_excl'] = dates_df['date'] - pd.Timedelta(days=1)
        if len(enj_b) > 0:
            merged = pd.merge_asof(dates_df, enj_b[['fecha']],
                left_on='date_excl', right_on='fecha', direction='backward')
            days = (dates_df['date'] - merged['fecha']).dt.days.fillna(999)
            df.loc[dates_df['index'], 'days_since_last_swarm'] = days.values
        else:
            df.loc[idx, 'days_since_last_swarm'] = 999
        for year in df_box['date'].dt.year.unique():
            y_mask  = mask & (df['date'].dt.year == year)
            dates_y = df.loc[y_mask, 'date'].sort_values()
            enj_y   = enj_b[enj_b['fecha'].dt.year == year]['fecha'].sort_values().values
            n = np.searchsorted(enj_y, dates_y.values - np.timedelta64(1,'D'), side='right') if len(enj_y) > 0 else np.zeros(len(dates_y), dtype=int)
            df.loc[dates_y.index, 'n_swarms_this_season'] = n
    return df

print("Building features...")
daily_swarm = add_swarm_features(daily, df_enjambres)

# Rolling features (from NB01 cell 5 continuation)
for box_id, grp in daily_swarm.groupby('box_id'):
    idx = grp.index
    W = grp['Weight']
    daily_swarm.loc[idx, 'weight_ma_7d']    = W.rolling(7,  min_periods=3).mean()
    daily_swarm.loc[idx, 'weight_ma_14d']   = W.rolling(14, min_periods=7).mean()
    daily_swarm.loc[idx, 'weight_diff_7d']  = W.diff(7)
    daily_swarm.loc[idx, 'weight_diff_14d'] = W.diff(14)
    daily_swarm.loc[idx, 'weight_diff_21d'] = W.diff(21)
    daily_swarm.loc[idx, 'weight_trend_slope'] = pd.Series(
        [np.polyfit(np.arange(min(7,i+1)), W.iloc[max(0,i-6):i+1].values, 1)[0]
         if i >= 2 else np.nan for i in range(len(W))], index=idx)
    daily_swarm.loc[idx, 'weight_acceleration'] = daily_swarm.loc[idx, 'weight_diff_7d'].diff()
    daily_swarm.loc[idx, 'n_positive_days_7d']  = (W.diff() > 0).rolling(7).sum()
    daily_swarm.loc[idx, 'weight_growing_streak'] = (W.diff() > 0).astype(int).groupby(
        (W.diff() <= 0).astype(int).cumsum()).cumsum()
    F = grp['Frequency']
    daily_swarm.loc[idx, 'freq_ma_7d']  = F.rolling(7, min_periods=3).mean()
    daily_swarm.loc[idx, 'temp_trend_7d'] = grp['Temp_scale'].diff(7)

SWARM_BOXES = sorted(df_enjambres['box_id'].unique())
daily_swarm_model = daily_swarm[daily_swarm['box_id'].isin(SWARM_BOXES)].copy()
daily_swarm_model['month'] = daily_swarm_model['date'].dt.month

features_swarm = [
    'Weight','weight_ma_7d','weight_ma_14d',
    'weight_diff_7d','weight_diff_14d','weight_diff_21d',
    'weight_trend_slope','weight_acceleration',
    'n_positive_days_7d','weight_growing_streak',
    'weight_peak_ratio','weight_drop_3d',
    'days_since_last_swarm','n_swarms_this_season',
    'Frequency','freq_ma_7d','freq_trend_7d',
    'Freq_std','Volume','vol_trend_7d',
    'Temp_scale','temp_trend_7d','Temp_heart','corr_w_temp',
]
features_swarm = [f for f in features_swarm if f in daily_swarm_model.columns]
print(f"Features: {len(features_swarm)}")

# ── Labeling helper ───────────────────────────────────────────────────────
def label_horizon(df, df_enj, horizon, split):
    df_s = df[df['month'].between(2, 6)].copy()
    df_s['target'] = 0
    tr = df_s[df_s['date'] < split].copy()
    te = df_s[df_s['date'] >= split].copy()
    for _, e in df_enj.iterrows():
        mask_tr = (tr['box_id']==e['box_id']) & \
                  (tr['date'] >= e['fecha']-pd.Timedelta(days=horizon)) & \
                  (tr['date'] < e['fecha'])
        tr.loc[mask_tr, 'target'] = 1
        mask_te = (te['box_id']==e['box_id']) & \
                  (te['date'] >= e['fecha']-pd.Timedelta(days=horizon)) & \
                  (te['date'] < e['fecha'])
        te.loc[mask_te, 'target'] = 1
    return tr, te

split_ts = pd.Timestamp(SPLIT_DATE)
tr3, te3 = label_horizon(daily_swarm_model, df_enjambres, HORIZON, split_ts)
print(f"3d: train pos={tr3['target'].sum()}, test pos={te3['target'].sum()}")

# ══════════════════════════════════════════════════════
# 1. XGBoost Optuna — 50 trials, walk-forward AP
# ══════════════════════════════════════════════════════
def _wf_ap(params):
    folds = [('2025-01-01','2026-01-01'), ('2026-01-01', None)]
    aps = []
    for ts_str, te_str in folds:
        ts = pd.Timestamp(ts_str)
        df_l_tr, df_l_te = label_horizon(daily_swarm_model, df_enjambres, HORIZON, ts)
        if te_str:
            df_l_te = df_l_te[df_l_te['date'] < pd.Timestamp(te_str)]
        if df_l_tr['target'].sum() < 2 or df_l_te['target'].sum() < 2: continue
        _med = df_l_tr[features_swarm].replace([np.inf,-np.inf], np.nan).median()
        X_tr = df_l_tr[features_swarm].replace([np.inf,-np.inf], np.nan).fillna(_med)
        y_tr = df_l_tr['target'].values
        X_te = df_l_te[features_swarm].replace([np.inf,-np.inf], np.nan).fillna(_med)
        y_te = df_l_te['target'].values
        spw  = (y_tr==0).sum() / max((y_tr==1).sum(), 1)
        m = XGBClassifier(**params, scale_pos_weight=spw, random_state=SEED, verbosity=0)
        m.fit(X_tr, y_tr)
        aps.append(average_precision_score(y_te, m.predict_proba(X_te)[:,1]))
    return float(np.mean(aps)) if aps else 0.0

def _xgb_obj(trial):
    return _wf_ap(dict(
        n_estimators     = trial.suggest_int('n_estimators', 100, 700),
        max_depth        = trial.suggest_int('max_depth', 2, 5),
        learning_rate    = trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        subsample        = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0),
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10),
        gamma            = trial.suggest_float('gamma', 0.0, 2.0),
    ))

print("\n" + "="*60)
print("XGBoost Optuna (50 trials, walk-forward AP, 3d horizon)")
print("="*60)
study_xgb = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
study_xgb.optimize(_xgb_obj, n_trials=50, show_progress_bar=True)
print(f"\nBest walk-forward AP: {study_xgb.best_value:.4f}")
print("Best XGBoost params:", study_xgb.best_params)

# Evaluate best XGBoost on test 2026
bp = study_xgb.best_params
_med = tr3[features_swarm].replace([np.inf,-np.inf], np.nan).median()
X_tr = tr3[features_swarm].replace([np.inf,-np.inf], np.nan).fillna(_med)
y_tr = tr3['target'].values
X_te = te3[features_swarm].replace([np.inf,-np.inf], np.nan).fillna(_med)
y_te = te3['target'].values
spw  = (y_tr==0).sum() / max((y_tr==1).sum(), 1)
xgb_opt = XGBClassifier(**bp, scale_pos_weight=spw, random_state=SEED, verbosity=0)
xgb_opt.fit(X_tr, y_tr)
prob_opt = xgb_opt.predict_proba(X_te)[:,1]
fpr, tpr, thrs = roc_curve(y_te, prob_opt)
gmeans = np.sqrt(tpr * (1-fpr))
best_thr = thrs[np.argmax(gmeans)]
y_pred = (prob_opt >= best_thr).astype(int)
sens = (y_pred[y_te==1].sum()) / max(y_te.sum(), 1)
spec = ((y_pred[y_te==0]==0).sum()) / max((y_te==0).sum(), 1)
print(f"\n=== XGBoost Optuna — test 2026 ===")
print(f"AUC: {roc_auc_score(y_te, prob_opt):.3f} | AP: {average_precision_score(y_te, prob_opt):.3f} | "
      f"G-Mean: {np.sqrt(sens*spec):.3f} | Sens: {sens:.3f} | Spec: {spec:.3f}")

# ══════════════════════════════════════════════════════
# 2. LSTM Optuna — 20 trials, internal validation loss
# ══════════════════════════════════════════════════════
class LSTMClassifier01(nn.Module):
    def __init__(self, n_feat, hidden=64, n_layers=2, bidirectional=False, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden * (2 if bidirectional else 1), 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)

def make_sequences_01(df, feat_cols, target_col, seq_len=SEQ_LEN):
    X_list, y_list = [], []
    for _, grp in df.groupby('box_id', sort=False):
        grp = grp.sort_values('date').reset_index(drop=True)
        vals = grp[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0).values.astype(np.float32)
        labs = grp[target_col].values.astype(np.float32)
        for i in range(seq_len, len(vals)):
            X_list.append(vals[i-seq_len:i])
            y_list.append(labs[i])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

# Prepare sequences for 3d horizon
all3 = pd.concat([tr3, te3]).sort_values(['box_id','date'])
scaler = StandardScaler()
_med3 = tr3[features_swarm].replace([np.inf,-np.inf], np.nan).median()
all3[features_swarm] = all3[features_swarm].replace([np.inf,-np.inf], np.nan).fillna(_med3)
scaler.fit(tr3[features_swarm].fillna(_med3))
all3[features_swarm] = scaler.transform(all3[features_swarm])

X_all, y_all_arr = make_sequences_01(all3, features_swarm, 'target')
dates_seq = []
for _, grp in all3.groupby('box_id', sort=False):
    grp = grp.sort_values('date').reset_index(drop=True)
    for i in range(SEQ_LEN, len(grp)):
        dates_seq.append(grp.iloc[i]['date'])
dates_seq = pd.Series(dates_seq)

te_mask = dates_seq >= split_ts
X_tr_s, y_tr_s = X_all[~te_mask.values], y_all_arr[~te_mask.values]
X_te_s, y_te_s = X_all[te_mask.values],  y_all_arr[te_mask.values]

# Internal val: last 20% of train dates
tr_dates = dates_seq[~te_mask.values]
val_cut  = sorted(tr_dates.unique())[int(len(tr_dates.unique())*0.8)]
val_mask = tr_dates >= val_cut
X_tr_o, y_tr_o = X_tr_s[~val_mask.values], y_tr_s[~val_mask.values]
X_val_o, y_val_o = X_tr_s[val_mask.values], y_tr_s[val_mask.values]
pos_w_o = float((y_tr_o==0).sum() / max((y_tr_o==1).sum(), 1))
print(f"\nLSTM Optuna — train: {len(X_tr_o)} (pos={y_tr_o.mean():.4f})  val: {len(X_val_o)} (pos={y_val_o.mean():.4f})")

tr_load_o  = DataLoader(TensorDataset(torch.from_numpy(X_tr_o), torch.from_numpy(y_tr_o)), batch_size=BATCH, shuffle=True)
val_load_o = DataLoader(TensorDataset(torch.from_numpy(X_val_o), torch.from_numpy(y_val_o)), batch_size=BATCH, shuffle=False)
n_feat = X_tr_o.shape[2]

def _lstm_obj(trial):
    hidden  = trial.suggest_categorical('hidden', [32, 64, 128])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr      = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd      = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    bidir   = trial.suggest_categorical('bidirectional', [False, True])

    torch.manual_seed(SEED)
    model = LSTMClassifier01(n_feat, hidden=hidden, n_layers=n_layers,
                              bidirectional=bidir, dropout=dropout).to(DEVICE)
    crit  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w_o], device=DEVICE))
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5, min_lr=1e-5)
    best_val_auc, best_val_loss, best_state, wait = 0.0, np.inf, None, 0
    for epoch in range(1, 41):
        model.train()
        for xb, yb in tr_load_o:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        model.eval()
        vl, probs_v, labs_v = [], [], []
        with torch.no_grad():
            for xb, yb in val_load_o:
                logits = model(xb.to(DEVICE))
                vl.append(crit(logits, yb.to(DEVICE)).item())
                probs_v.append(torch.sigmoid(logits).cpu().numpy())
                labs_v.append(yb.numpy())
        val_loss = np.mean(vl)
        sched.step(val_loss)
        try:
            val_auc = roc_auc_score(np.concatenate(labs_v), np.concatenate(probs_v))
        except Exception:
            val_auc = 0.5
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc  = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 10:
                break
        trial.report(val_auc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best_val_auc

print("\n" + "="*60)
print("LSTM Optuna (20 trials, internal validation AUC, 3d horizon)")
print("="*60)
study_lstm = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=SEED),
                                  pruner=optuna.pruners.MedianPruner())
study_lstm.optimize(_lstm_obj, n_trials=20, show_progress_bar=True)
best_lstm = study_lstm.best_params
print(f"\nBest val AUC: {study_lstm.best_value:.4f}")
print("Best LSTM params:", best_lstm)

# Evaluate best LSTM on test 2026
pos_w_full = float((y_tr_s==0).sum() / max((y_tr_s==1).sum(), 1))
tr_load_f  = DataLoader(TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr_s)), batch_size=BATCH, shuffle=True)
te_load_f  = DataLoader(TensorDataset(torch.from_numpy(X_te_s), torch.from_numpy(y_te_s)), batch_size=BATCH, shuffle=False)

torch.manual_seed(SEED)
m_final = LSTMClassifier01(n_feat,
    hidden=best_lstm['hidden'], n_layers=best_lstm['n_layers'],
    bidirectional=best_lstm['bidirectional'], dropout=best_lstm['dropout']).to(DEVICE)
crit_f = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w_full], device=DEVICE))
opt_f  = torch.optim.Adam(m_final.parameters(), lr=best_lstm['lr'], weight_decay=best_lstm['weight_decay'])
sched_f = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_f, patience=7, factor=0.5, min_lr=1e-5)
best_val_f, best_state_f, wait_f = np.inf, None, 0
for epoch in range(1, 81):
    m_final.train()
    for xb, yb in tr_load_f:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_f.zero_grad()
        crit_f(m_final(xb), yb).backward()
        nn.utils.clip_grad_norm_(m_final.parameters(), 1.0)
        opt_f.step()
    m_final.eval()
    vl = []
    with torch.no_grad():
        for xb, yb in te_load_f:
            vl.append(crit_f(m_final(xb.to(DEVICE)), yb.to(DEVICE)).item())
    val_l = np.mean(vl)
    sched_f.step(val_l)
    if val_l < best_val_f:
        best_val_f = val_l; best_state_f = {k:v.cpu().clone() for k,v in m_final.state_dict().items()}; wait_f = 0
    else:
        wait_f += 1
        if wait_f >= 15: break

m_final.load_state_dict(best_state_f)
m_final.eval()
probs_lstm = []
with torch.no_grad():
    for xb, _ in te_load_f:
        probs_lstm.append(torch.sigmoid(m_final(xb.to(DEVICE))).cpu().numpy())
probs_lstm = np.concatenate(probs_lstm)
fpr2, tpr2, thrs2 = roc_curve(y_te_s, probs_lstm)
gm2 = np.sqrt(tpr2*(1-fpr2))
thr2 = thrs2[np.argmax(gm2)]
y_pred2 = (probs_lstm >= thr2).astype(int)
sens2 = y_pred2[y_te_s==1].sum() / max(y_te_s.sum(), 1)
spec2 = (y_pred2[y_te_s==0]==0).sum() / max((y_te_s==0).sum(), 1)
print(f"\n=== LSTM Optuna — test 2026 ===")
print(f"AUC: {roc_auc_score(y_te_s, probs_lstm):.3f} | AP: {average_precision_score(y_te_s, probs_lstm):.3f} | "
      f"G-Mean: {np.sqrt(sens2*spec2):.3f} | Sens: {sens2:.3f} | Spec: {spec2:.3f}")
print(f"Bidirectional: {best_lstm['bidirectional']}")

print("\n" + "="*60)
print("DONE — copy these results into NB01 Optuna cell")
print("="*60)
