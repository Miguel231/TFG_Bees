"""
Swarm-risk inference pipeline — replicates the exact feature engineering and
LSTM architecture from notebooks/models/03_swarm_night_enhanced.ipynb (the best
model overall: AUC=0.887, 13/14 test events detected, 218 false alarms).

Required artifacts (export them by running the notebook's Section 7.3,
"Exportar el modelo para inferencia", then copy the api_export/ folder here
as MODEL_DIR — see README.md):
    lstm_uni_03.pt       — trained LSTM weights (state_dict)
    scaler_03.pkl         — StandardScaler fit on the training split
    feat_a_03.json        — exact feature list, in the order the model expects
    median_fill_03.json   — per-feature train medians, for NaN imputation
    model_meta_03.json    — architecture + window hyperparameters
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


class LSTMClassifier(nn.Module):
    """Must match notebooks/models/03_swarm_night_enhanced.ipynb exactly."""

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


class SwarmPredictor:
    """Loads all exported artifacts once and exposes .predict(df_raw)."""

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self._load_artifacts()

    def _load_artifacts(self):
        missing = [
            f for f in (
                "lstm_uni_03.pt", "scaler_03.pkl", "feat_a_03.json",
                "median_fill_03.json", "model_meta_03.json",
            )
            if not os.path.exists(os.path.join(self.model_dir, f))
        ]
        if missing:
            raise FileNotFoundError(
                f"Faltan artefactos del modelo en {self.model_dir}: {missing}. "
                "Ejecuta la Sección 7.3 de 03_swarm_night_enhanced.ipynb y copia "
                "el contenido de api_export/ a esta carpeta (ver README.md)."
            )

        with open(os.path.join(self.model_dir, "model_meta_03.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(os.path.join(self.model_dir, "feat_a_03.json"), encoding="utf-8") as f:
            self.feat_a = json.load(f)
        with open(os.path.join(self.model_dir, "median_fill_03.json"), encoding="utf-8") as f:
            self.median_fill = pd.Series(json.load(f))

        self.scaler = joblib.load(os.path.join(self.model_dir, "scaler_03.pkl"))

        self.model = LSTMClassifier(
            n_feat=self.meta["n_features"],
            hidden=self.meta["hidden"],
            n_layers=self.meta["n_layers"],
            bidirectional=self.meta["bidirectional"],
        )
        state = torch.load(
            os.path.join(self.model_dir, "lstm_uni_03.pt"), map_location=DEVICE,
        )
        self.model.load_state_dict(state)
        self.model.to(DEVICE)
        self.model.eval()

        self.seq_len = self.meta["seq_len"]
        self.hour_s = self.meta["morning_hour_start"]
        self.hour_e = self.meta["morning_hour_end"]

    # ------------------------------------------------------------------
    # Feature engineering — mirrors notebooks/models/03_swarm_night_enhanced.ipynb
    # cells 6, 8, 10 (resample, daily/window/night aggregation, rolling features).
    # ------------------------------------------------------------------
    def build_features(self, raw: pd.DataFrame, box_id) -> pd.DataFrame:
        """
        raw: dataframe with the unified-CSV columns for ONE hive, covering at
        least `seq_len` + 21 days of raw 15-min-ish readings (more history =
        more reliable rolling features). Columns expected (same as the CSV):
        'Hive name', 'Time', 'Weight', 'Frequency', 'Volume',
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

        # 15-min resample + bounded forward-fill (gaps > 2h stay NaN)
        df15 = (
            df.set_index("Time")[
                ["Weight", "Frequency", "Volume", "Temp_scale", "Hum_scale", "Temp_heart", "Hum_heart"]
            ]
            .resample("15min").mean()
            .ffill(limit=8)
            .reset_index()
        )
        df15["box_id"] = box_id

        def _remove_spikes(s, thr=5.0):
            spike = s.diff().abs() > thr
            return s.where(~spike)
        df15["Weight"] = _remove_spikes(df15["Weight"])
        df15["date"] = df15["Time"].dt.normalize()

        # Daily aggregates
        daily = df15.groupby(["box_id", "date"]).agg(
            Weight_mean=("Weight", "mean"), Weight_min=("Weight", "min"),
            Weight_max=("Weight", "max"), Weight_std=("Weight", "std"),
            Freq_mean=("Frequency", "mean"), Freq_max=("Frequency", "max"),
            Freq_std=("Frequency", "std"), Vol_mean=("Volume", "mean"),
            Vol_max=("Volume", "max"), Temp_mean=("Temp_scale", "mean"),
            Temp_max=("Temp_scale", "max"),
        ).reset_index()

        # Morning window (10-14h)
        morning = df15[df15["Time"].dt.hour.between(self.hour_s, self.hour_e - 1)]
        win = morning.groupby(["box_id", "date"]).agg(
            win_Weight_first=("Weight", "first"), win_Weight_last=("Weight", "last"),
            win_Weight_min=("Weight", "min"), win_Weight_max=("Weight", "max"),
            win_Weight_std=("Weight", "std"), win_Freq_mean=("Frequency", "mean"),
            win_Freq_max=("Frequency", "max"), win_Freq_std=("Frequency", "std"),
            win_Vol_mean=("Volume", "mean"), win_Vol_max=("Volume", "max"),
        ).reset_index()
        win["win_Weight_drop"] = win["win_Weight_first"] - win["win_Weight_last"]
        win["win_Weight_range"] = win["win_Weight_max"] - win["win_Weight_min"]

        feat_day = daily.merge(win, on=["box_id", "date"], how="left").sort_values("date")

        # Night features (22h-7h) — from RAW (non-ffilled) readings, no contamination
        night_raw = df.copy()
        night_raw["Weight"] = night_raw["Weight"].where(night_raw["Weight"].between(3, 120), np.nan)
        night_raw["_date"] = night_raw["Time"].dt.normalize()
        late = night_raw["Time"].dt.hour >= 22
        night_raw["_night_date"] = night_raw["_date"]
        night_raw.loc[late, "_night_date"] = night_raw.loc[late, "_date"] + pd.Timedelta(days=1)
        night_raw = night_raw[(night_raw["Time"].dt.hour >= 22) | (night_raw["Time"].dt.hour <= 7)]
        night_agg = night_raw.groupby(["box_id", "_night_date"]).agg(
            night_Freq_mean=("Frequency", "mean"), night_Freq_max=("Frequency", "max"),
            night_Freq_std=("Frequency", "std"), night_Weight_mean=("Weight", "mean"),
            night_Vol_mean=("Volume", "mean"), night_Vol_max=("Volume", "max"),
        ).reset_index().rename(columns={"_night_date": "date"})

        prev_eve = (
            df[df["Time"].dt.hour.between(18, 21)]
            .assign(date=lambda d: d["Time"].dt.normalize() + pd.Timedelta(days=1))
            .groupby(["box_id", "date"])["Weight"].mean()
            .reset_index().rename(columns={"Weight": "_eve_w"})
        )
        early_morn = (
            df[df["Time"].dt.hour.between(6, 9)]
            .assign(date=lambda d: d["Time"].dt.normalize())
            .groupby(["box_id", "date"])["Weight"].mean()
            .reset_index().rename(columns={"Weight": "_morn_w"})
        )
        wdelta = prev_eve.merge(early_morn, on=["box_id", "date"], how="inner")
        wdelta["night_weight_delta"] = wdelta["_morn_w"] - wdelta["_eve_w"]
        wdelta = wdelta[["box_id", "date", "night_weight_delta"]]

        feat_day = feat_day.merge(night_agg, on=["box_id", "date"], how="left")
        feat_day = feat_day.merge(wdelta, on=["box_id", "date"], how="left")

        # Rolling features (per-hive, sorted by date)
        feat_day = feat_day.sort_values("date").reset_index(drop=True)
        W, F, T = feat_day["Weight_mean"], feat_day["Freq_mean"], feat_day["Temp_mean"]
        feat_day["w_ma7"] = W.rolling(7, min_periods=3).mean()
        feat_day["w_ma14"] = W.rolling(14, min_periods=5).mean()
        feat_day["w_diff7"] = W - W.shift(7)
        feat_day["w_diff14"] = W - W.shift(14)
        feat_day["weight_std_roll14"] = W.rolling(14, min_periods=7).std()
        feat_day["f_ma7"] = F.rolling(7, min_periods=3).mean()
        feat_day["f_diff7"] = F - F.shift(7)
        feat_day["corr_w_temp"] = W.rolling(14, min_periods=7).corr(T)
        freq_med7 = feat_day["Freq_max"].rolling(7, min_periods=3).median().shift(1)
        feat_day["freq_spike_ratio"] = feat_day["win_Freq_max"] / freq_med7.where(freq_med7 >= 10)

        NF, NFmax = feat_day["night_Freq_mean"], feat_day["night_Freq_max"]
        feat_day["night_f_ma7"] = NF.rolling(7, min_periods=3).mean()
        feat_day["night_fmax_ma7"] = NFmax.rolling(7, min_periods=3).mean()
        feat_day["night_fmax_ma3"] = NFmax.rolling(3, min_periods=2).mean()
        nmed7 = NFmax.rolling(7, min_periods=3).median().shift(1)
        feat_day["night_freq_spike_ratio"] = NFmax / nmed7.where(nmed7 >= 10)
        feat_day["night_to_morning_freq"] = feat_day["win_Freq_max"] / NFmax.where(NFmax >= 10)
        feat_day["night_wdelta_ma3"] = feat_day["night_weight_delta"].rolling(3, min_periods=2).mean()

        # days_since_swarm / has_prior_swarm: no swarm history available live ->
        # sentinel (999 / 0), same convention as training for "no prior swarm known"
        feat_day["days_since_swarm"] = 999
        feat_day["has_prior_swarm"] = 0
        feat_day["day_of_year"] = feat_day["date"].dt.dayofyear

        return feat_day.replace([np.inf, -np.inf], np.nan)

    def predict(self, raw: pd.DataFrame, box_id) -> dict:
        """Returns the swarm-risk probability for the most recent available day."""
        feat_day = self.build_features(raw, box_id)
        if len(feat_day) < self.seq_len:
            raise ValueError(
                f"Se necesitan al menos {self.seq_len} dias de historia tras el "
                f"resample; hay {len(feat_day)}. Aporta mas dias de datos crudos."
            )

        X = feat_day[self.feat_a].fillna(self.median_fill)
        X = self.scaler.transform(X)
        seq = X[-self.seq_len:]  # most recent seq_len days
        seq_t = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(seq_t)).item()

        last_date = feat_day["date"].iloc[-1]
        risk_level = "ALTO" if prob >= 0.5 else ("MEDIO" if prob >= 0.2 else "BAJO")
        return {
            "box_id": box_id,
            "date": str(last_date.date()),
            "swarm_risk_probability": round(prob, 4),
            "risk_level": risk_level,
            "horizon_days": self.meta["horizon_days"],
            "message": (
                f"Riesgo de enjambrazon en los proximos {self.meta['horizon_days']} dias: "
                f"{risk_level} ({prob*100:.1f}%)"
            ),
        }
