"""
Bee Monitor — Swarm Risk + Honey Super Dashboard.
Data updated automatically every Monday at 9 am via n8n.

Run:
    streamlit run dashboard.py --server.port 8501
"""
import base64
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Bee Monitor — UAB TFG",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme injection ─────────────────────────────────────────────────────────────
def _b64_img(path: Path) -> str:
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except Exception:
        return ""

_UAB_LOGO = _b64_img(Path(__file__).parent / "static" / "uab_logo.png")

# Seamless pointy-top honeycomb tile (49×84 px, R=28)
_HONEYCOMB_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='49' height='84'>"
    "<polygon points='24.5,0 48.75,14 48.75,42 24.5,56 0.25,42 0.25,14' "
    "fill='none' stroke='%23FFC700' stroke-width='1.4' stroke-opacity='0.055'/>"
    "<polygon points='0,42 24.25,56 24.25,84 0,98 -24.25,84 -24.25,56' "
    "fill='none' stroke='%23FFC700' stroke-width='1.4' stroke-opacity='0.055'/>"
    "<polygon points='49,42 73.25,56 73.25,84 49,98 24.75,84 24.75,56' "
    "fill='none' stroke='%23FFC700' stroke-width='1.4' stroke-opacity='0.055'/>"
    "</svg>"
)

def inject_theme():
    logo_html = (
        f'<img id="uab-corner" src="data:image/png;base64,{_UAB_LOGO}" alt="UAB"/>'
        if _UAB_LOGO else
        '<div id="uab-corner" style="font-size:18px;font-weight:700;color:#003087;'
        'letter-spacing:2px;">UAB</div>'
    )
    st.markdown(f"""
    <style>
    /* ── Honeycomb background ─────────────────────────────────────── */
    .stApp {{
        background-image: url("data:image/svg+xml,{_HONEYCOMB_SVG}");
        background-repeat: repeat;
        background-attachment: fixed;
        background-size: 49px 84px;
    }}
    /* ── UAB logo — fixed bottom-right ───────────────────────────── */
    #uab-corner {{
        position: fixed;
        bottom: 18px;
        right: 20px;
        z-index: 99999;
        width: 108px;
        background: rgba(255,255,255,0.93);
        border-radius: 8px;
        padding: 6px 10px 5px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.35);
        opacity: 0.90;
        transition: opacity .2s;
    }}
    #uab-corner:hover {{ opacity: 1; }}
    </style>
    {logo_html}
    """, unsafe_allow_html=True)

inject_theme()

# ── Constants ──────────────────────────────────────────────────────────────────
SWARM_HIVES  = [1, 2, 3, 4, 5, 8, 13, 14]
RESULTS_FILE = Path(__file__).parent / "latest_results.json"
HISTORY_FILE = Path(__file__).parent / "run_history.json"
ALZA_7D      = Path(__file__).parent / "models" / "lgbm_alza_solo_hive_7d.pkl"
ALZA_14D     = Path(__file__).parent / "models" / "lgbm_alza_solo_hive_14d.pkl"
FEAT_CSV     = Path(__file__).resolve().parent.parent.parent / "daily_features_final.csv"

# Hives with registered alza history — must match the training dict in train_alza_features.py
_ALZA_DICT = {
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
ALZA_HIVES = sorted({hid for hives in _ALZA_DICT.values()
                      for hid, delta in hives.items() if delta > 0})

RISK_HEX   = {"HIGH":"#FF4B4B","MEDIUM":"#FFA500","LOW":"#21C55D",
               "NO_DATA":"#666666","ERROR":"#444444",
               "ALTO":"#FF4B4B","MEDIO":"#FFA500","BAJO":"#21C55D","SIN_DATOS":"#666666"}
RISK_EMOJI = {"HIGH":"🔴","MEDIUM":"🟡","LOW":"🟢","NO_DATA":"⚪","ERROR":"⚫",
               "ALTO":"🔴","MEDIO":"🟡","BAJO":"🟢","SIN_DATOS":"⚪"}
RISK_ORDER = {"HIGH":0,"ALTO":0,"MEDIUM":1,"MEDIO":1,"LOW":2,"BAJO":2,
               "NO_DATA":3,"SIN_DATOS":3,"ERROR":4}
_ES_TO_EN  = {"ALTO":"HIGH","MEDIO":"MEDIUM","BAJO":"LOW","SIN_DATOS":"NO_DATA","ERROR":"ERROR"}

_now      = datetime.now()
IN_SEASON = 3 <= _now.month <= 6


def to_en(risk: str) -> str:
    return _ES_TO_EN.get(risk, risk)


def next_monday_9am() -> datetime:
    now = datetime.now()
    days_ahead = (7 - now.weekday()) % 7
    if days_ahead == 0 and now.hour >= 9:
        days_ahead = 7
    return (now + timedelta(days=days_ahead)).replace(
        hour=9, minute=0, second=0, microsecond=0)


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_results():
    if not RESULTS_FILE.exists():
        return None, "no_data"
    try:
        data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        for p in data.get("predictions", []):
            p["risk_level"] = to_en(p.get("risk_level", "ERROR"))
        data["alerts"]      = [p for p in data["predictions"] if p.get("risk_level") == "HIGH"]
        data["alert_count"] = len(data["alerts"])
        return data, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=60)
def load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        hist = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        for entry in hist:
            for p in entry.get("predictions", []):
                p["risk_level"] = to_en(p.get("risk_level", "ERROR"))
        return hist
    except Exception:
        return []


@st.cache_data(ttl=10)
def check_api(api_url: str):
    try:
        return requests.get(f"{api_url}/health", timeout=3).json()
    except Exception:
        return None


@st.cache_data(ttl=300)
def load_feat_csv():
    df = pd.read_csv(str(FEAT_CSV), parse_dates=["date"])
    return df.sort_values(["box_id", "date"])


@st.cache_resource
def load_alza_7d():
    import joblib
    return joblib.load(ALZA_7D) if ALZA_7D.exists() else None


@st.cache_resource
def load_alza_14d():
    import joblib
    return joblib.load(ALZA_14D) if ALZA_14D.exists() else None


# ── UI helpers ─────────────────────────────────────────────────────────────────
def hive_card(box_id: int, pred, delta=None) -> str:
    if pred is None:
        risk, prob, pct_text, pct_val = "NO_DATA", None, "—", 0
    else:
        risk     = to_en(pred.get("risk_level", "ERROR"))
        prob     = pred.get("swarm_risk_probability")
        pct_text = f"{prob * 100:.1f}%" if prob is not None else "—"
        pct_val  = min(int((prob or 0) * 100), 100)

    c     = RISK_HEX.get(risk, "#666")
    emoji = RISK_EMOJI.get(risk, "?")

    delta_html = ""
    if delta is not None:
        sign  = "▲" if delta > 0 else "▼"
        dcol  = "#FF4B4B" if delta > 0.05 else ("#21C55D" if delta < -0.05 else "#888")
        delta_html = (
            f'<div style="font-size:10px;color:{dcol};margin-bottom:6px;">'
            f'{sign} {abs(delta)*100:.1f}% vs last run</div>'
        )

    return f"""
    <div style="border:1px solid {c}55;border-left:4px solid {c};
                border-radius:10px;padding:18px 16px 14px;background:{c}12;margin-bottom:4px;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">
            <span style="font-size:11px;font-weight:600;letter-spacing:1.5px;color:#aaa;">HIVE {box_id}</span>
            <span style="font-size:20px;line-height:1;">{emoji}</span>
        </div>
        <div style="font-size:38px;font-weight:700;color:{c};line-height:1;margin-bottom:4px;">{pct_text}</div>
        <div style="font-size:10px;font-weight:700;letter-spacing:2px;color:{c};margin-bottom:8px;">{risk}</div>
        {delta_html}
        <div style="background:#2a2a2a;border-radius:4px;height:5px;overflow:hidden;">
            <div style="background:{c};height:100%;width:{pct_val}%;border-radius:4px;"></div>
        </div>
    </div>"""


def alza_card(box_id: int, prob, threshold: float, data_date=None) -> str:
    """Honey super card with threshold-relative coloring."""
    if prob is None:
        c, emoji, label, pct, bar_w = "#666666", "⚪", "NO DATA", "—", 0
    elif prob >= threshold:
        c, emoji, label = "#21C55D", "✅", "ADD SUPER"
        pct   = f"{prob*100:.0f}%"
        bar_w = min(int(prob * 100), 100)
    elif prob >= threshold * 0.5:
        c, emoji, label = "#FFA500", "🟡", "MONITOR"
        pct   = f"{prob*100:.0f}%"
        bar_w = min(int(prob * 100), 100)
    else:
        c, emoji, label = "#888888", "⏳", "WAIT"
        pct   = f"{prob*100:.0f}%"
        bar_w = min(int(prob * 100), 100)

    date_html = ""
    if data_date is not None:
        days_ago = (_now.date() - pd.Timestamp(data_date).date()).days
        stale    = days_ago > 14
        dcol     = "#FF6B6B" if stale else "#aaa"
        date_html = f'<div style="font-size:9px;color:{dcol};margin-top:4px;">data: {data_date} ({days_ago}d ago)</div>'

    return f"""
    <div style="border:1px solid {c}55;border-left:4px solid {c};
                border-radius:10px;padding:14px 12px 10px;background:{c}12;margin-bottom:4px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:11px;font-weight:600;letter-spacing:1.5px;color:#aaa;">HIVE {box_id}</span>
            <span style="font-size:18px;">{emoji}</span>
        </div>
        <div style="font-size:28px;font-weight:700;color:{c};line-height:1.1;margin-bottom:2px;">{pct}</div>
        <div style="font-size:10px;font-weight:700;letter-spacing:2px;color:{c};margin-bottom:6px;">{label}</div>
        <div style="background:#2a2a2a;border-radius:4px;height:4px;overflow:hidden;">
            <div style="background:{c};height:100%;width:{bar_w}%;border-radius:4px;"></div>
        </div>
        {date_html}
    </div>"""


def run_alza_model(bundle, feat_df):
    """Return (list of (hive_id, prob, data_date), threshold) for hives with alza history."""
    if bundle is None:
        return [], 0.50
    clf   = bundle["model"]
    feats = bundle["features"]
    med   = bundle["median"]
    thr   = bundle.get("threshold", 0.50)
    out   = []
    # Only predict for hives with registered alza history (same set used in training)
    available = set(feat_df["box_id"].unique())
    for hid in ALZA_HIVES:
        if hid not in available:
            continue  # hive not yet in feature CSV — skip silently
        hdf = feat_df[feat_df["box_id"] == hid]
        if hdf.empty:
            out.append((hid, None, None))
            continue
        row = hdf.sort_values("date").tail(1)
        data_date = str(row["date"].iloc[0].date())
        try:
            X    = row[feats].replace([np.inf, -np.inf], np.nan).fillna(med)
            prob = float(clf.predict_proba(X)[:, 1][0])
            out.append((hid, prob, data_date))
        except Exception:
            out.append((hid, None, data_date))
    return out, thr


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐝 Bee Monitor")
    st.caption("TFG — UAB · Miguel Arpa · 2026")
    st.divider()

    api_url = st.text_input(
        "API URL", value="http://localhost:8000", label_visibility="collapsed"
    ).rstrip("/")

    health = check_api(api_url)
    if health is None:
        st.error("API offline")
    elif health.get("status") == "ok":
        st.success("API online ✓")
    else:
        st.warning("Model not loaded")

    st.divider()

    next_run = next_monday_9am()
    delta_h  = int((next_run - _now).total_seconds() // 3600)
    st.markdown(
        f"**Next auto-analysis**\n\n"
        f"📅 {next_run.strftime('%a %d/%m/%Y')} · 9:00 am\n\n"
        f"⏳ in ~{delta_h}h\n\n"
        f"{'🌸 Swarm season active' if IN_SEASON else '❄️ Off-season'}"
    )

    st.divider()

    if st.button("🔄 Refresh", use_container_width=True):
        load_results.clear()
        load_history.clear()
        load_feat_csv.clear()
        st.rerun()

    st.divider()
    st.markdown(
        "**Swarm risk:**\n\n"
        "🔴 HIGH ≥ 50%\n\n"
        "🟡 MEDIUM 20–50%\n\n"
        "🟢 LOW < 20%\n\n"
        "**Honey super:**\n\n"
        "✅ ADD SUPER — prob ≥ threshold\n\n"
        "🟡 MONITOR — prob ≥ 50% of threshold\n\n"
        "⏳ WAIT — prob < 50% of threshold"
    )
    st.divider()
    st.caption(
        "**Models:**\n\n"
        "🐝 Swarm: LSTM Uni · AUC 0.887\n\n"
        "🍯 Alza 7d: LightGBM · AUC 0.765 · CV 0.782±0.041\n\n"
        "🍯 Alza 14d: XGBoost · AUC 0.701 · CV 0.716±0.052\n\n"
        "Auto-update: n8n · Mon 9:00 am"
    )

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🐝 Bee Monitor — IoT Sensor Dashboard")
st.caption(
    "Swarm risk prediction + honey super recommendations · "
    "IoT sensors (weight, frequency, temperature) · "
    "Auto-updated every **Monday at 9:00 am** via n8n"
)

data, err = load_results()
history   = load_history()

if err == "offline":
    st.error("**API offline.** `cd TFG_Bees/api && python -m uvicorn main:app --port 8000`")
    st.stop()

prev_map: dict[int, float] = {}
if len(history) >= 2:
    for p in history[1].get("predictions", []):
        prob = p.get("swarm_risk_probability")
        if prob is not None:
            prev_map[int(p.get("box_id", -1))] = prob

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_swarm, tab_alza, tab_sensor, tab_trends, tab_models, tab_hist = st.tabs([
    "🐝 Swarm Risk",
    "🍯 Honey Supers",
    "📊 Sensor Data",
    "📈 Risk Trends",
    "🔬 Models",
    "📜 History",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Swarm Risk
# ══════════════════════════════════════════════════════════════════════════════
with tab_swarm:
    if err == "no_data" or data is None:
        st.info(
            f"**No swarm predictions yet.** First auto-analysis: "
            f"**{next_run.strftime('%A %d/%m/%Y')} at 9:00 am** via n8n.\n\n"
            "Monitored hives: " + ", ".join(f"Hive {h}" for h in SWARM_HIVES)
        )
        st.divider()
        for chunk in [SWARM_HIVES[:4], SWARM_HIVES[4:]]:
            cols = st.columns(4)
            for i, hid in enumerate(chunk):
                with cols[i]:
                    st.html(hive_card(hid, None))
    else:
        preds    = data.get("predictions", [])
        alerts   = data.get("alerts", [])
        run_date = data.get("run_date", "?")
        run_ts   = data.get("run_timestamp", run_date)

        if alerts:
            hives_str = "  ·  ".join(f"Hive {a['box_id']}" for a in alerts)
            st.error(
                f"⚠️ **{len(alerts)} hive(s) at HIGH swarming risk** "
                f"within the next 3 days\n\n🔴 {hives_str}"
            )
        else:
            st.success(f"✅ No HIGH-risk alerts · analysis from {run_date}")

        prev_alert = (sum(1 for p in history[1].get("predictions", [])
                          if to_en(p.get("risk_level","")) == "HIGH")
                      if len(history) >= 2 else None)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Last analysis", run_date)
        m2.metric("Hives monitored", len(preds))
        m3.metric("⚠️ HIGH alerts", len(alerts),
                  delta=len(alerts) - prev_alert if prev_alert is not None else None,
                  delta_color="inverse")
        all_probs  = [p.get("swarm_risk_probability") or 0 for p in preds
                      if p.get("swarm_risk_probability") is not None]
        peak_prob  = max(all_probs) if all_probs else None
        peak_label = ("HIGH" if (peak_prob or 0) >= 0.50
                      else ("MEDIUM" if (peak_prob or 0) >= 0.20 else "LOW"))
        m4.metric("Peak risk", f"{peak_prob*100:.1f}% ({peak_label})" if peak_prob else "—")

        st.divider()
        if IN_SEASON:
            st.info("🌸 **Swarm season (Mar–Jun)** — heightened vigilance recommended.")

        st.markdown(f"### Hive status · {run_date}")
        pred_map = {int(p.get("box_id", -1)): p for p in preds}

        for chunk in [SWARM_HIVES[:4], SWARM_HIVES[4:]]:
            cols = st.columns(4)
            for i, hid in enumerate(chunk):
                with cols[i]:
                    pred  = pred_map.get(hid)
                    prob  = pred.get("swarm_risk_probability") if pred else None
                    delta = (prob - prev_map[hid]) if (prob is not None and hid in prev_map) else None
                    st.html(hive_card(hid, pred, delta=delta))

        st.divider()
        st.markdown("### Summary table")
        rows = []
        for p in sorted(preds, key=lambda x: RISK_ORDER.get(x.get("risk_level","ERROR"), 9)):
            risk  = p.get("risk_level", "ERROR")
            prob  = p.get("swarm_risk_probability")
            hid   = int(p.get("box_id", 0))
            delta = (prob - prev_map[hid]) if (prob is not None and hid in prev_map) else None
            rows.append({
                "": RISK_EMOJI.get(risk, "?"),
                "Hive": hid,
                "Risk level": risk,
                "Probability": f"{prob*100:.1f}%" if prob is not None else "—",
                "vs prev": (f"{'▲' if delta>0 else '▼'} {abs(delta)*100:.1f}%"
                             if delta is not None else "—"),
                "Data up to": p.get("date", "—"),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            csv_b = pd.DataFrame(rows).to_csv(index=False).encode()
            st.download_button("⬇ Download CSV", csv_b,
                               file_name=f"swarm_risk_{run_date}.csv", mime="text/csv")

        st.divider()
        c1, c2 = st.columns([3, 1])
        with c1:
            st.caption(f"🤖 Auto-analysis via n8n · Mon 9:00 am · "
                       f"LSTM Uni · 3-day horizon · AUC = 0.887 · Last run: {run_ts}")
        with c2:
            st.caption(f"Next: {next_run.strftime('%d/%m %H:%M')}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Honey Supers
# ══════════════════════════════════════════════════════════════════════════════
with tab_alza:
    st.markdown("### Honey Super Placement Recommendations")

    if not ALZA_7D.exists() and not ALZA_14D.exists():
        st.warning("No alza models found. Run `python train_alza_features.py` from `TFG_Bees/api/`.")
    elif not FEAT_CSV.exists():
        st.warning("Feature CSV not found. Run the weekly pipeline first.")
    else:
        try:
            bundle_7d  = load_alza_7d()
            bundle_14d = load_alza_14d()
            feat_df    = load_feat_csv()
            last_date  = feat_df["date"].max()

            days_stale = (_now.date() - last_date.date()).days
            freshness  = ("🟢 fresh" if days_stale <= 3
                          else "🟡 slightly stale" if days_stale <= 10
                          else "🔴 stale")
            st.caption(
                f"Data up to **{last_date.date()}** ({days_stale} days ago) · {freshness} · "
                f"Thresholds G-Mean optimized · Validated 3-fold walk-forward CV"
            )

            res_7d,  thr_7d  = run_alza_model(bundle_7d,  feat_df)
            res_14d, thr_14d = run_alza_model(bundle_14d, feat_df)

            # ── Model captions ─────────────────────────────────────────────────
            def _cap(b):
                if b is None: return "—"
                cv_s = (f" · CV {b['cv_mean']:.3f}±{b['cv_std']:.3f}"
                        if b.get("cv_mean") is not None else "")
                return (f"{b.get('model_type','?')} · {b.get('config','?')} · "
                        f"AUC {b['auc']:.3f} · G-Mean {b.get('gmean',0):.3f} · "
                        f"thr {b.get('threshold',0):.3f} · "
                        f"detected {b['detected']}/20{cv_s}")

            # ── Summary banner ─────────────────────────────────────────────────
            add_7d  = [hid for hid, p, _ in res_7d  if p is not None and p >= thr_7d]
            add_14d = [hid for hid, p, _ in res_14d if p is not None and p >= thr_14d]
            if add_7d or add_14d:
                msg = ""
                if add_7d:  msg += f"**7d ADD SUPER:** hives {', '.join(map(str, add_7d))}  "
                if add_14d: msg += f"**14d ADD SUPER:** hives {', '.join(map(str, add_14d))}"
                st.success(msg.strip())
            else:
                st.info("No honey super placements recommended at this time.")

            # ── Side-by-side 7d / 14d ─────────────────────────────────────────
            col_l, col_r = st.columns(2)

            def show_alza_col(col, results, thr, horizon_label, cap_str):
                with col:
                    st.markdown(f"#### {horizon_label} horizon")
                    st.caption(cap_str)
                    n_hives  = len(results)
                    n_cols   = 4
                    for row_start in range(0, n_hives, n_cols):
                        chunk = results[row_start:row_start + n_cols]
                        cols  = st.columns(len(chunk))
                        for j, (hid, prob, d_date) in enumerate(chunk):
                            with cols[j]:
                                st.html(alza_card(hid, prob, thr, d_date))
                    # Probability table (expandable)
                    with st.expander("📋 Full probability table"):
                        tbl = []
                        for hid, prob, d_date in sorted(results,
                                key=lambda x: -(x[1] or -1)):
                            if prob is None:
                                status = "NO DATA"
                            elif prob >= thr:
                                status = "✅ ADD"
                            elif prob >= thr * 0.5:
                                status = "🟡 MONITOR"
                            else:
                                status = "⏳ WAIT"
                            tbl.append({
                                "Hive": hid,
                                "Probability": f"{prob*100:.1f}%" if prob is not None else "—",
                                "Status": status,
                                "Data date": d_date or "—",
                            })
                        st.dataframe(pd.DataFrame(tbl), use_container_width=True,
                                     hide_index=True)

            show_alza_col(col_l, res_7d,  thr_7d,  "7-day",  _cap(bundle_7d))
            show_alza_col(col_r, res_14d, thr_14d, "14-day", _cap(bundle_14d))

        except Exception as exc:
            st.error(f"Error running alza model: {exc}")
            import traceback
            st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Sensor Data
# ══════════════════════════════════════════════════════════════════════════════
with tab_sensor:
    st.markdown("### Sensor Data — Recent Trends")

    if not FEAT_CSV.exists():
        st.warning("Feature CSV not found.")
    else:
        feat_df   = load_feat_csv()
        all_hives = sorted(feat_df["box_id"].unique())

        # Controls
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            sel_hives = st.multiselect(
                "Hives to display",
                options=all_hives,
                default=all_hives[:8],
                format_func=lambda x: f"Hive {x}",
            )
        with c2:
            n_days = st.selectbox("Period", [14, 30, 60, 90], index=1,
                                  format_func=lambda x: f"Last {x} days")
        with c3:
            sensor = st.selectbox("Sensor", ["Weight (kg)", "Frequency (Hz)", "Temperature (°C)"])

        sensor_col = {
            "Weight (kg)":       "Weight",
            "Frequency (Hz)":    "Frequency",
            "Temperature (°C)":  "Temp_scale",
        }[sensor]

        if not sel_hives:
            st.info("Select at least one hive.")
        elif sensor_col not in feat_df.columns:
            st.warning(f"Column `{sensor_col}` not in feature CSV.")
        else:
            cutoff = feat_df["date"].max() - pd.Timedelta(days=n_days)
            df_plot = (feat_df[
                (feat_df["box_id"].isin(sel_hives)) &
                (feat_df["date"] >= cutoff)
            ][["date", "box_id", sensor_col]]
              .dropna(subset=[sensor_col]))

            if df_plot.empty:
                st.warning("No data for selected hives/period.")
            else:
                # Pivot: rows=date, cols=hive
                pivot = (df_plot.pivot_table(index="date", columns="box_id",
                                              values=sensor_col, aggfunc="mean")
                         .rename(columns=lambda h: f"Hive {h}"))
                st.line_chart(pivot, use_container_width=True, height=380)
                st.caption(
                    f"{sensor} · last {n_days} days · "
                    f"data up to {feat_df['date'].max().date()}"
                )

                # Latest values table
                st.divider()
                st.markdown("#### Latest readings")
                latest_rows = []
                for hid in sel_hives:
                    hdf = feat_df[feat_df["box_id"] == hid]
                    if hdf.empty:
                        continue
                    last_row  = hdf.sort_values("date").tail(1).iloc[0]
                    val       = last_row.get(sensor_col)
                    prev_rows = hdf.sort_values("date").tail(8)
                    trend_val = None
                    if len(prev_rows) >= 7 and sensor_col in prev_rows.columns:
                        vals = prev_rows[sensor_col].dropna().values
                        if len(vals) >= 2:
                            trend_val = float(vals[-1] - vals[0])
                    trend_str = ("—" if trend_val is None
                                 else f"▲ +{trend_val:.2f}" if trend_val > 0
                                 else f"▼ {trend_val:.2f}")
                    latest_rows.append({
                        "Hive": hid,
                        sensor: f"{val:.2f}" if val is not None else "—",
                        "7-day trend": trend_str,
                        "Data date": str(last_row["date"].date()),
                    })
                if latest_rows:
                    st.dataframe(pd.DataFrame(latest_rows),
                                 use_container_width=True, hide_index=True)

                # Weight-specific stats
                if sensor_col == "Weight":
                    st.divider()
                    st.markdown("#### Weight statistics (selected period)")
                    stat_rows = []
                    for hid in sel_hives:
                        sub = df_plot[df_plot["box_id"] == hid]["Weight"] if "Weight" in df_plot else df_plot[df_plot["box_id"] == hid][sensor_col]
                        if sub.empty:
                            continue
                        stat_rows.append({
                            "Hive": hid,
                            "Min (kg)":  f"{sub.min():.1f}",
                            "Max (kg)":  f"{sub.max():.1f}",
                            "Mean (kg)": f"{sub.mean():.1f}",
                            "Change (kg)": f"{sub.iloc[-1]-sub.iloc[0]:+.1f}" if len(sub) > 1 else "—",
                        })
                    if stat_rows:
                        st.dataframe(pd.DataFrame(stat_rows),
                                     use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Risk Trends
# ══════════════════════════════════════════════════════════════════════════════
with tab_trends:
    st.markdown("### Swarm Risk Probability Over Time")
    st.caption("One data point per weekly n8n run · each line = one hive")

    if not history:
        st.info("No history yet. Trends will appear after the first automated weekly run.")
    else:
        trend_rows = []
        for entry in reversed(history):
            row = {"date": entry.get("run_date", "?")}
            pm  = {int(p.get("box_id", -1)): p.get("swarm_risk_probability")
                   for p in entry.get("predictions", [])}
            for hid in SWARM_HIVES:
                row[f"Hive {hid}"] = (pm[hid] * 100) if (hid in pm and pm[hid] is not None) else None
            trend_rows.append(row)

        df_trend  = pd.DataFrame(trend_rows).set_index("date")
        hive_cols = list(df_trend.columns)
        selected  = st.multiselect("Hives", options=hive_cols, default=hive_cols)

        if selected:
            st.line_chart(df_trend[selected], use_container_width=True, height=350)
            st.caption("🔴 HIGH ≥ 50%  |  🟡 MEDIUM 20–50%  |  🟢 LOW < 20%  (y-axis = %)")

            st.divider()
            st.markdown("### Risk distribution across all runs")
            dist_rows = []
            for entry in history:
                for p in entry.get("predictions", []):
                    dist_rows.append({"Hive": f"Hive {p.get('box_id')}",
                                      "Risk": to_en(p.get("risk_level", "ERROR"))})
            if dist_rows:
                df_dist = pd.DataFrame(dist_rows)
                pivot   = (df_dist.groupby(["Hive", "Risk"])
                           .size().reset_index(name="Count")
                           .pivot(index="Hive", columns="Risk", values="Count")
                           .fillna(0).astype(int))
                col_order = [c for c in ["HIGH","MEDIUM","LOW","NO_DATA","ERROR"]
                             if c in pivot.columns]
                st.dataframe(pivot[col_order], use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Models
# ══════════════════════════════════════════════════════════════════════════════
with tab_models:
    st.markdown("### Model Performance Summary")
    st.caption(
        "All models trained on data from 2023 to end of 2025 · "
        "tested on 2026 · thresholds optimized by G-Mean"
    )

    # ── Swarm model ────────────────────────────────────────────────────────────
    st.markdown("#### 🐝 Swarm Prediction Model")
    st.markdown(
        "**Architecture:** LSTM Unidirectional · 2 layers · hidden=64 · seq_len=14 · "
        "44 features (morning 10–14h + night 22–7h windows) · trained with `BCEWithLogitsLoss`\n\n"
        "**Data:** 15-minute sensor readings resampled to daily morning/night aggregates · "
        "8 hives with swarm history · Feb–Jun filter"
    )
    swarm_data = {
        "Model":       ["LSTM Uni ★ DEPLOYED", "LSTM Bi",      "XGBoost baseline"],
        "AUC-ROC":     [0.887,                  0.789,          0.717],
        "Avg Prec":    [0.160,                  0.092,          0.141],
        "G-Mean":      [0.844,                  0.743,          0.715],
        "Sensitivity": [0.925,                  0.700,          0.650],
        "Specificity": [0.771,                  0.789,          0.787],
        "Det/14":      ["13/14",                "10/14",        "10/14"],
        "False Alarms":[218,                    200,            202],
    }
    df_swarm = pd.DataFrame(swarm_data)
    st.dataframe(df_swarm, use_container_width=True, hide_index=True)
    st.caption(
        "Det/14 = swarm events detected in 2026 test set (14 total) · "
        "False Alarms = predictions outside any swarm window"
    )

    st.divider()

    # ── Alza models ────────────────────────────────────────────────────────────
    st.markdown("#### 🍯 Honey Super Placement Models")
    st.markdown(
        "**Feature search:** 6 configurations × 2 algorithms × 2 horizons = 24 models · "
        "best selected by G-Mean · validated with 3-fold walk-forward CV"
    )

    alza_data = {
        "Model":         ["LGBM bio_no_tmp 7d ★", "XGB acoustic_26 14d ★",
                          "XGB hive_19 14d",        "LGBM hive_19 7d (baseline)"],
        "Horizon":       ["7 days",    "14 days",   "14 days",  "7 days"],
        "Features":      [46,          26,           19,         19],
        "AUC":           [0.765,       0.701,        0.710,      0.606],
        "G-Mean":        [0.730,       0.718,        0.729,      0.641],
        "Sensitivity":   [0.80,        0.67,         0.81,       0.58],
        "Specificity":   [0.66,        0.77,         0.66,       0.71],
        "Det/20":        ["14/20",     "15/20",      "15/20",    "11/20"],
        "False Alarms":  [597,         377,          565,        516],
        "CV AUC":        ["0.782±0.041","0.716±0.052","—",       "—"],
        "Threshold":     [0.083,       0.257,        0.308,      0.069],
    }
    df_alza = pd.DataFrame(alza_data)
    st.dataframe(df_alza, use_container_width=True, hide_index=True)
    st.caption(
        "★ = deployed in dashboard · "
        "Det/20 = 2026 ADD interventions detected in horizon window · "
        "CV AUC = walk-forward cross-validation (3 folds: val2024, val2025, test2026)"
    )

    st.divider()

    # ── Feature importance ─────────────────────────────────────────────────────
    st.markdown("#### Top features — deployed models")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**7d LGBM bio_no_tmp** (top 10)")
        fi_7d = {
            "Feature": ["apiary_weight_mean","n_hives_active","apiary_weight_std",
                        "Temp_scale","Weight_max","weight_ma_14d","weight_std_7d",
                        "freq_ma_7d","apiary_growth_std","weight_historical_std"],
            "Importance": [19,17,9,6,5,4,4,4,4,3],
            "Type": ["Apiary","Apiary","Apiary","Env","Weight","Weight","Weight",
                     "Acoustic","Apiary","Weight"],
        }
        if ALZA_7D.exists():
            try:
                import joblib
                b7 = joblib.load(ALZA_7D)
                clf7 = b7["model"]
                fi_series = pd.Series(clf7.feature_importances_,
                                      index=b7["features"]).sort_values(ascending=False)
                fi_7d = {
                    "Feature":    list(fi_series.head(10).index),
                    "Importance": [round(v, 4) for v in fi_series.head(10).values],
                }
            except Exception:
                pass
        st.dataframe(pd.DataFrame(fi_7d), use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("**14d XGB acoustic_26** (top 10)")
        fi_14d = {
            "Feature": ["sin_dayofyear","cos_dayofyear","weight_ma_7d","Freq_std",
                        "Temp_scale","days_since_last_alza","weight_diff_14d",
                        "Weight","Volume","days_in_season"],
            "Importance": [0.114,0.097,0.073,0.061,0.057,0.053,0.049,0.048,0.047,0.045],
        }
        if ALZA_14D.exists():
            try:
                import joblib
                b14  = joblib.load(ALZA_14D)
                clf14 = b14["model"]
                fi_series14 = pd.Series(clf14.feature_importances_,
                                        index=b14["features"]).sort_values(ascending=False)
                fi_14d = {
                    "Feature":    list(fi_series14.head(10).index),
                    "Importance": [round(v, 4) for v in fi_series14.head(10).values],
                }
            except Exception:
                pass
        st.dataframe(pd.DataFrame(fi_14d), use_container_width=True, hide_index=True)

    st.divider()

    # ── Walk-forward CV ────────────────────────────────────────────────────────
    st.markdown("#### Walk-forward Cross-Validation")
    st.caption(
        "3 temporal folds · train always precedes validation · "
        "ensures models generalise to future data"
    )
    cv_data = {
        "Model":            ["LGBM bio_no_tmp 7d", "XGB acoustic_26 14d"],
        "Fold 1 (val 2024)":[0.743,                 0.786],
        "Fold 2 (val 2025)":[0.838,                 0.661],
        "Fold 3 (test 2026)":[0.765,                0.701],
        "Mean AUC":         [0.782,                 0.716],
        "Std":              [0.041,                 0.052],
        "Assessment":       ["Robust — consistent across folds",
                             "Stable — 2025 lower (different season)"],
    }
    st.dataframe(pd.DataFrame(cv_data), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — History
# ══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.markdown("### n8n Automated Run History")
    st.caption("One entry per weekly analysis · newest first")

    if not history:
        st.info(
            "No history yet. Each Monday at 9 am, n8n triggers a full pipeline "
            "run and the result is logged here automatically."
        )
    else:
        st.markdown(f"**{len(history)} run(s) recorded** · n8n · every Monday 9:00 am")

        summary = []
        for entry in history:
            preds_e = entry.get("predictions", [])
            rc = {k: 0 for k in ["HIGH","MEDIUM","LOW","NO_DATA","ERROR"]}
            for p in preds_e:
                k = to_en(p.get("risk_level","ERROR"))
                rc[k] = rc.get(k, 0) + 1
            alert_ids = entry.get("alerts", [])
            summary.append({
                "Date":  entry.get("run_date", "?"),
                "Time":  entry.get("run_timestamp", "?")[11:16]
                         if len(entry.get("run_timestamp","")) >= 16 else "?",
                "Hives": len(preds_e),
                "🔴 HIGH":   rc.get("HIGH", 0),
                "🟡 MEDIUM": rc.get("MEDIUM", 0),
                "🟢 LOW":    rc.get("LOW", 0),
                "Alert hives": ", ".join(str(a) for a in alert_ids) if alert_ids else "—",
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Risk evolution heatmap")
        st.caption("Columns = runs (most recent first) · Rows = hives")
        heat_rows = []
        for hid in SWARM_HIVES:
            row = {"Hive": f"Hive {hid}"}
            for entry in history:
                label = entry.get("run_date", "?")
                pm    = {int(p.get("box_id",-1)): p for p in entry.get("predictions",[])}
                p     = pm.get(hid)
                if p is None:
                    row[label] = "—"
                else:
                    risk = to_en(p.get("risk_level","?"))
                    prob = p.get("swarm_risk_probability")
                    pct  = f"{prob*100:.0f}%" if prob is not None else "?"
                    row[label] = f"{RISK_EMOJI.get(risk,'?')} {pct}"
            heat_rows.append(row)
        st.dataframe(pd.DataFrame(heat_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Run details")
        for entry in history:
            run_d    = entry.get("run_date", "?")
            run_t    = entry.get("run_timestamp", run_d)
            n_alerts = entry.get("alert_count", 0)
            icon     = "🔴" if n_alerts > 0 else "✅"
            with st.expander(f"{icon} {run_d}  ·  {n_alerts} alert(s)", expanded=False):
                det = []
                for p in sorted(entry.get("predictions", []),
                                 key=lambda x: RISK_ORDER.get(
                                     to_en(x.get("risk_level","ERROR")), 9)):
                    risk = to_en(p.get("risk_level","?"))
                    prob = p.get("swarm_risk_probability")
                    det.append({
                        "": RISK_EMOJI.get(risk,"?"),
                        "Hive": int(p.get("box_id",0)),
                        "Risk": risk,
                        "Probability": f"{prob*100:.1f}%" if prob is not None else "—",
                        "Data up to": p.get("date","—"),
                    })
                if det:
                    st.dataframe(pd.DataFrame(det), use_container_width=True, hide_index=True)
                st.caption(f"Run at: {run_t}")
