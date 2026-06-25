"""
Swarm-risk inference pipeline — NB01 (multi-horizon daily feature set).

Two models are loaded:
  • 3-day  alert: LSTM Uni  (G-Mean=0.866, 13/14 events, FA=136)
  • 14-day alert: LSTM Bi   (G-Mean=0.815, 13/14 events, FA=55)

Required artifacts in MODEL_DIR (api/models/):
    lstm_uni_01_3d.pt       — NB01 LSTM Uni 3d weights
    lstm_bi_01_14d.pt       — NB01 LSTM Bi 14d weights
    scaler_01.pkl           — StandardScaler fitted on training split
    median_fill_01.json     — per-feature train medians for NaN imputation
    model_meta_01_3d.json   — architecture params for 3d model
    model_meta_01_14d.json  — architecture params for 14d model
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = [
    "Weight", "weight_ma_7d", "weight_ma_14d",
    "weight_diff_7d", "weight_diff_14d", "weight_diff_21d",
    "weight_trend_slope", "weight_acceleration",
    "n_positive_days_7d", "weight_growing_streak",
    "weight_peak_ratio", "weight_drop_3d",
    "days_since_last_swarm", "n_swarms_this_season",
    "Frequency", "freq_ma_7d", "freq_trend_7d",
    "Freq_std", "Volume", "vol_trend_7d",
    "Temp_scale", "temp_trend_7d", "Temp_heart", "corr_w_temp",
]


def _slope(arr):
    v = arr[~np.isnan(arr)]
    return float(np.polyfit(np.arange(len(v)), v, 1)[0]) if len(v) >= 5 else np.nan


class LSTMClassifier(nn.Module):
    def __init__(self, n_feat, hidden=64, n_layers=2, bidirectional=False, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            n_feat, hidden, num_layers=n_layers, batch_first=True,
            bidirectional=bidirectional, dropout=dropout if n_layers > 1 else 0.0,
        )
        factor = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden * factor), nn.Dropout(dropout),
            nn.Linear(hidden * factor, 32), nn.ReLU(),
            nn.Dropout(dropout * 0.7), nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(1)


def _load_lstm(model_dir: str, weights_file: str, meta_file: str) -> tuple:
    with open(os.path.join(model_dir, meta_file), encoding="utf-8") as f:
        meta = json.load(f)
    model = LSTMClassifier(
        n_feat=meta["n_features"],
        hidden=meta["hidden"],
        n_layers=meta["n_layers"],
        bidirectional=meta["bidirectional"],
        dropout=meta.get("dropout", 0.3),
    )
    state = torch.load(os.path.join(model_dir, weights_file), map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, meta


class SwarmPredictor:
    """Loads NB01 3d + 14d models and exposes .predict(df_raw, box_id)."""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self._load_artifacts()

    def _load_artifacts(self):
        required = [
            "lstm_uni_01_3d.pt", "lstm_bi_01_14d.pt",
            "scaler_01.pkl", "median_fill_01.json",
            "model_meta_01_3d.json", "model_meta_01_14d.json",
        ]
        missing = [f for f in required if not os.path.exists(os.path.join(self.model_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"Faltan artefactos en {self.model_dir}: {missing}"
            )

        self.scaler = joblib.load(os.path.join(self.model_dir, "scaler_01.pkl"))
        with open(os.path.join(self.model_dir, "median_fill_01.json"), encoding="utf-8") as f:
            self.median_fill = pd.Series(json.load(f))

        self.model_3d,  self.meta_3d  = _load_lstm(self.model_dir, "lstm_uni_01_3d.pt",  "model_meta_01_3d.json")
        self.model_14d, self.meta_14d = _load_lstm(self.model_dir, "lstm_bi_01_14d.pt",  "model_meta_01_14d.json")
        self.seq_len = self.meta_3d["seq_len"]  # 21 for both

    def build_features(self, raw: pd.DataFrame, box_id) -> pd.DataFrame:
        """
        raw: DataFrame with unified-CSV columns for one or more hives.
        Returns daily feature table for box_id with the 24 NB01 features.
        Columns expected: 'Hive name', 'Time', 'Weight', 'Frequency', 'Volume',
        'Temperature heart', 'Humidity heart', 'Temperature scale', 'Humidity scale'.
        """
        df = raw.rename(columns={
            "Hive name": "box_id",
            "Temperature heart": "Temp_heart",
            "Humidity heart": "Hum_heart",
            "Temperature scale": "Temp_scale",
            "Humidity scale": "Hum_scale",
        }).copy()
        df["Time"] = pd.to_datetime(df["Time"], format="mixed", dayfirst=False)
        df = df[df["box_id"] == box_id].sort_values("Time")
        df["Weight"] = df["Weight"].where(df["Weight"].between(3, 120), np.nan)

        # 15-min resample + forward-fill (max 2h gap)
        df15 = (
            df.set_index("Time")[
                ["Weight", "Frequency", "Volume", "Temp_scale", "Temp_heart", "Hum_heart"]
            ]
            .resample("15min").mean()
            .ffill(limit=8)
            .reset_index()
        )
        df15["box_id"] = box_id

        # Spike filter on weight
        spike = df15["Weight"].diff().abs() > 5.0
        df15["Weight"] = df15["Weight"].where(~spike)
        df15["date"] = df15["Time"].dt.normalize()

        # Daily aggregates
        daily = df15.groupby(["box_id", "date"]).agg(
            Weight=("Weight", "mean"),
            Frequency=("Frequency", "mean"),
            Freq_std=("Frequency", "std"),
            Volume=("Volume", "mean"),
            Temp_scale=("Temp_scale", "mean"),
            Temp_heart=("Temp_heart", "mean"),
        ).reset_index().sort_values("date").reset_index(drop=True)

        # Rolling weight features
        W = daily["Weight"]
        daily["weight_ma_7d"]       = W.rolling(7,  min_periods=3).mean()
        daily["weight_ma_14d"]      = W.rolling(14, min_periods=5).mean()
        daily["weight_diff_7d"]     = W - W.shift(7)
        daily["weight_diff_14d"]    = W - W.shift(14)
        daily["weight_diff_21d"]    = W - W.shift(21)
        daily["weight_diff_1d"]     = W.diff(1)
        daily["weight_trend_slope"] = W.rolling(14, min_periods=5).apply(_slope, raw=True)
        daily["weight_acceleration"]= daily["weight_diff_1d"].diff(1)
        daily["n_positive_days_7d"] = (daily["weight_diff_1d"] > 0).rolling(7, min_periods=3).sum()

        # weight_growing_streak — current consecutive days of positive gain
        ganando = (daily["weight_diff_1d"] > 0).astype(int)
        streak  = ganando.groupby((ganando != ganando.shift()).cumsum()).cumsum()
        daily["weight_growing_streak"] = streak.values

        # weight_peak_ratio — weight vs expanding 95th pct
        exp_peak = W.expanding(min_periods=14).quantile(0.95).shift(1)
        daily["weight_peak_ratio"] = W / (exp_peak + 1e-6)

        # weight_drop_3d
        daily["weight_drop_3d"] = -(W.diff(3))

        # Acoustic rolling
        F = daily["Frequency"]
        daily["freq_ma_7d"]   = F.rolling(7, min_periods=3).mean()
        daily["freq_trend_7d"]= F.diff(7)

        # Volume trend
        daily["vol_trend_7d"] = daily["Volume"].diff(7)

        # Temperature trend
        daily["temp_trend_7d"] = daily["Temp_scale"].diff(7)

        # 14-day rolling Pearson correlation Weight vs Temp_scale
        daily["corr_w_temp"] = W.rolling(14, min_periods=7).corr(daily["Temp_scale"])

        # Swarm history: no live history available → sentinel values
        daily["days_since_last_swarm"] = 999
        daily["n_swarms_this_season"]  = 0

        return daily.replace([np.inf, -np.inf], np.nan)

    def _run_model(self, feat_day: pd.DataFrame, model: LSTMClassifier, seq_len: int) -> float:
        X = feat_day[FEATURES].fillna(self.median_fill)
        X = self.scaler.transform(X)
        seq = X[-seq_len:]
        seq_t = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return torch.sigmoid(model(seq_t)).item()

    def predict(self, raw: pd.DataFrame, box_id) -> dict:
        """Returns swarm-risk for the most recent day at both 3d and 14d horizons."""
        feat_day = self.build_features(raw, box_id)
        if len(feat_day) < self.seq_len:
            raise ValueError(
                f"Se necesitan al menos {self.seq_len} dias de historia; "
                f"hay {len(feat_day)}. Aporta mas dias de datos crudos."
            )

        last_date = feat_day["date"].iloc[-1]

        p3  = self._run_model(feat_day, self.model_3d,  self.meta_3d["seq_len"])
        p14 = self._run_model(feat_day, self.model_14d, self.meta_14d["seq_len"])

        def _risk(p):
            return "ALTO" if p >= 0.5 else ("MEDIO" if p >= 0.2 else "BAJO")

        risk_3d  = _risk(p3)
        risk_14d = _risk(p14)

        # Primary alert based on 3d model (most actionable)
        return {
            "box_id":                   box_id,
            "date":                     str(last_date.date()),
            "risk_level":               risk_3d,
            "swarm_risk_probability":   round(p3, 4),
            "horizon_days":             3,
            "risk_14d": {
                "probability":  round(p14, 4),
                "risk_level":   risk_14d,
                "horizon_days": 14,
            },
            "message": (
                f"Riesgo 3d: {risk_3d} ({p3*100:.1f}%)  |  "
                f"Riesgo 14d: {risk_14d} ({p14*100:.1f}%)"
            ),
        }
