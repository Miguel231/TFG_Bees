"""
update_alza_features.py — Weekly alza feature CSV updater.

Steps:
  1. Download last N days from beehivemonitoring.com (all 17 alza hives, Playwright)
  2. Append new rows to the master raw CSV (deduplicate by hive+time)
  3. Rebuild daily_features_final.csv from scratch using the notebook pipeline
  4. Output JSON status to stdout (n8n reads it)

Run manually:
    python update_alza_features.py [--days 30] [--no-download]

n8n weekly cron:
    python update_alza_features.py --days 30
"""
import argparse
import asyncio
import io
import json
import os
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE      = Path(__file__).parent
DATA_DIR  = HERE.parent.parent / "TFG_Bees" / "data"   # fallback
# Try to resolve relative to script location
_data_try = HERE.parent / "data"
if _data_try.exists():
    DATA_DIR = _data_try

RAW_CSV   = DATA_DIR / "12062026all_boxes.csv"
FEAT_CSV  = HERE.parent.parent / "daily_features_prod.csv"
FEAT_CSV2 = DATA_DIR / "daily_features_prod.csv"        # mirror copy in data/

SHARE_URL = os.environ.get(
    "BEEHIVE_SHARE_URL",
    "https://main.beehivemonitoring.com/c36f58c6b327462fa1b23da7f652697d",
)

# All hive prefixes with registered alza history (same as ALZA_HIVES in dashboard)
ALZA_HIVE_PREFIXES = [
    "001","002","003","004","005","006","007","008",
    "009","010","011","012","013","014","015","016","017"
]

# ── Alza ground truth (update manually when new alzas are added) ───────────────
_ALZAS = {
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

def _build_df_alzas():
    rows = []
    for fecha, cols in _ALZAS.items():
        for bid, delta in cols.items():
            rows.append({
                "fecha":  pd.to_datetime(fecha),
                "box_id": int(bid),
                "accion": "ADD" if delta > 0 else "REMOVE",
            })
    return pd.DataFrame(rows).sort_values(["box_id","fecha"]).reset_index(drop=True)


# ── Step 1 — Download from beehivemonitoring.com ───────────────────────────────
def _fmt_date(dt: datetime) -> str:
    return f"{dt.month}/{dt.day}/{dt.year}"


async def download_excel(days: int = 30) -> bytes:
    from playwright.async_api import async_playwright
    today     = datetime.today()
    date_from = _fmt_date(today - timedelta(days=days))
    date_to   = _fmt_date(today)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx     = await browser.new_context(accept_downloads=True)
        page    = await ctx.new_page()

        print(f"[alza] Opening {SHARE_URL} ...", file=sys.stderr)
        await page.goto(SHARE_URL, wait_until="networkidle", timeout=90_000)

        await page.locator("button").filter(has_text="menu").last.click()
        await page.wait_for_selector("text=Export Excel", timeout=8_000)
        await page.get_by_text("Export Excel").click()
        await page.wait_for_selector("text=Date from", timeout=8_000)

        date_from_input = page.locator("input").first
        await date_from_input.click(click_count=3)
        await date_from_input.fill(date_from)
        await page.keyboard.press("Tab")

        date_to_input = page.locator("input").nth(1)
        await date_to_input.click(click_count=3)
        await date_to_input.fill(date_to)
        await page.keyboard.press("Tab")

        print(f"[alza] Range: {date_from} → {date_to}", file=sys.stderr)

        # Uncheck all checkboxes
        all_cb = page.locator('mat-checkbox, input[type="checkbox"]')
        n = await all_cb.count()
        for i in range(n):
            cb    = all_cb.nth(i)
            inner = cb.locator('input[type="checkbox"]')
            if await inner.count() > 0:
                if await inner.is_checked():
                    await inner.click(force=True)
            else:
                if await cb.is_checked():
                    await cb.click(force=True)

        # Select all 17 alza hives
        checked = 0
        for prefix in ALZA_HIVE_PREFIXES:
            row = page.locator(f'mat-checkbox:has-text("{prefix}"), label:has-text("{prefix}")')
            if await row.count() > 0:
                inner = row.first.locator('input[type="checkbox"]')
                if await inner.count() > 0:
                    await inner.click(force=True)
                else:
                    await row.first.click()
                checked += 1
                print(f"  [✓] {prefix}", file=sys.stderr)
            else:
                print(f"  [?] {prefix} not found", file=sys.stderr)

        print(f"[alza] {checked}/{len(ALZA_HIVE_PREFIXES)} hives selected", file=sys.stderr)

        async with page.expect_download(timeout=60_000) as dl_info:
            await page.get_by_role("button", name="Download excel").click()
        download = await dl_info.value
        path     = await download.path()
        data     = Path(path).read_bytes()
        print(f"[alza] Downloaded: {len(data):,} bytes", file=sys.stderr)

        await ctx.close()
        await browser.close()
        return data


def xlsx_to_raw_df(xlsx_bytes: bytes) -> pd.DataFrame:
    """Parse xlsx exported from beehivemonitoring.com.

    Uses direct zipfile/XML parsing so it handles files where the exporter
    omits xl/sharedStrings.xml (openpyxl raises KeyError in that case).
    Supports both shared-string and inline-string cell types.
    """
    import xml.etree.ElementTree as ET

    _NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"

    def _tag(name: str) -> str:
        return f"{{{_NS}}}{name}"

    with zipfile.ZipFile(io.BytesIO(xlsx_bytes), "r") as z:
        names = z.namelist()

        # --- Shared strings (may be absent) ---
        shared: list = []
        if "xl/sharedStrings.xml" in names:
            ss_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in ss_root.iter(_tag("si")):
                parts = [t.text or "" for t in si.iter(_tag("t"))]
                shared.append("".join(parts))

        # --- Find first worksheet ---
        sheet_path = next(
            (n for n in sorted(names)
             if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")),
            None,
        )
        if sheet_path is None:
            raise ValueError("No worksheet found in xlsx")

        ws_root = ET.fromstring(z.read(sheet_path))

    # --- Parse cells row by row ---
    _EPOCH = pd.Timestamp("1899-12-30")

    def _cell_value(cell):
        ct = cell.get("t", "")
        v_el  = cell.find(_tag("v"))
        is_el = cell.find(_tag("is"))
        if ct == "s":
            # Shared string reference
            try:
                return shared[int(v_el.text)] if v_el is not None else None
            except (IndexError, TypeError, ValueError):
                return None
        if ct == "inlineStr":
            if is_el is not None:
                t_el = is_el.find(_tag("t"))
                return t_el.text if t_el is not None else None
            return None
        if ct == "b":
            return bool(int(v_el.text)) if v_el is not None else None
        if v_el is None or v_el.text is None:
            return None
        # Numeric / date-serial
        try:
            num = float(v_el.text)
        except ValueError:
            return v_el.text
        # Detect date style: integers up to ~60000 could be dates; use style attr
        s_attr = cell.get("s")
        if s_attr is not None:
            # Date serials for 2020–2030 are roughly 43831–49640; heuristic
            if 40000 < num < 60000:
                return (_EPOCH + pd.Timedelta(days=num)).to_pydatetime()
        return num

    # Map column letter(s) → 0-based index
    def _col_idx(ref: str) -> int:
        letters = "".join(c for c in ref if c.isalpha())
        idx = 0
        for ch in letters:
            idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
        return idx - 1

    raw_rows: list[list] = []
    for row_el in ws_root.iter(_tag("row")):
        cells_in_row: dict[int, object] = {}
        for cell in row_el.iter(_tag("c")):
            r_attr = cell.get("r", "")
            if r_attr:
                ci = _col_idx(r_attr)
                cells_in_row[ci] = _cell_value(cell)
        if cells_in_row:
            max_ci = max(cells_in_row)
            raw_rows.append([cells_in_row.get(i) for i in range(max_ci + 1)])

    if not raw_rows:
        raise ValueError("Empty xlsx")

    max_len = max(len(r) for r in raw_rows)
    for r in raw_rows:
        r.extend([None] * (max_len - len(r)))

    headers = [str(h).strip() if h is not None else "" for h in raw_rows[0]]
    df      = pd.DataFrame(raw_rows[1:], columns=headers)

    keep = ["Hive name","Time","Temperature heart","Humidity heart",
            "Frequency","Volume","Temperature scale","Humidity scale","Weight"]
    df   = df[[c for c in keep if c in df.columns]].copy()

    df["Time"] = pd.to_datetime(
        df["Time"].astype(str).str.replace(" "," ").str.replace(" "," "),
        format="%m/%d/%Y %I:%M:%S %p", errors="coerce",
    )
    df["Hive name"] = df["Hive name"].astype(str).str.extract(r"^0*(\d+)")[0].astype(int)
    df = df.dropna(subset=["Time"]).sort_values(["Hive name","Time"]).reset_index(drop=True)
    # Only keep alza hives
    df = df[df["Hive name"].between(1, 17)]
    return df


# ── Step 2 — Merge new rows with master raw CSV ───────────────────────────────
def merge_raw(new_df: pd.DataFrame) -> pd.DataFrame:
    print(f"[alza] New rows: {len(new_df):,}", file=sys.stderr)
    if not RAW_CSV.exists():
        print(f"[alza] Raw CSV not found, creating from new data", file=sys.stderr)
        out = new_df.rename(columns={"Hive name":"Hive name"})
        return out

    print(f"[alza] Loading raw CSV ({RAW_CSV.name}) ...", file=sys.stderr)
    old_df = pd.read_csv(str(RAW_CSV), parse_dates=["Time"],
                         low_memory=False)
    if "Hive name" not in old_df.columns and "box_id" in old_df.columns:
        old_df = old_df.rename(columns={"box_id": "Hive name"})

    # Align columns
    common = [c for c in old_df.columns if c in new_df.columns]
    combined = pd.concat([old_df[common], new_df[common]], ignore_index=True)
    before   = len(combined)
    combined = combined.drop_duplicates(subset=["Hive name","Time"]).sort_values(
        ["Hive name","Time"]).reset_index(drop=True)
    print(f"[alza] Merged: {before:,} → {len(combined):,} rows "
          f"(+{len(combined)-len(old_df):,} new)", file=sys.stderr)
    return combined


def save_raw(df: pd.DataFrame):
    df.to_csv(str(RAW_CSV), index=False)
    print(f"[alza] Saved raw CSV: {RAW_CSV}", file=sys.stderr)


# ── Step 3 — Feature pipeline (extracted from 04_honey_super.ipynb) ────────────
def build_daily_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw = df_raw.rename(columns={"Hive name": "box_id"}) \
                   if "Hive name" in df_raw.columns else df_raw.copy()
    df_raw["Time"] = pd.to_datetime(df_raw["Time"])
    df_raw = df_raw.sort_values(["box_id","Time"]).reset_index(drop=True)

    df_scale = df_raw.dropna(subset=["Weight"]).copy()
    df_heart = df_raw.dropna(subset=["Frequency"]).copy()

    df_scale = df_scale[(df_scale["Weight"] >= 10) & (df_scale["Weight"] <= 120)]

    # IQR filter per hive
    cleaned = []
    for bid, grp in df_scale.groupby("box_id"):
        Q1, Q3 = grp["Weight"].quantile(0.25), grp["Weight"].quantile(0.75)
        IQR = Q3 - Q1
        cleaned.append(grp[(grp["Weight"] >= Q1-5*IQR) & (grp["Weight"] <= Q3+5*IQR)])
    df_scale = pd.concat(cleaned) if cleaned else df_scale

    daily_scale = (df_scale
        .groupby(["box_id", df_scale["Time"].dt.date])
        .agg(Weight=("Weight","mean"), Weight_max=("Weight","max"),
             Weight_min=("Weight","min"), Temp_scale=("Temperature scale","mean"),
             Humidity_scale=("Humidity scale","mean"))
        .reset_index().rename(columns={"Time":"date"}))

    daily_heart = (df_heart
        .groupby(["box_id", df_heart["Time"].dt.date])
        .agg(Frequency=("Frequency","mean"), Freq_std=("Frequency","std"),
             Temp_heart=("Temperature heart","mean"),
             Humidity_heart=("Humidity heart","mean"), Volume=("Volume","mean"))
        .reset_index().rename(columns={"Time":"date"}))

    daily = daily_scale.merge(daily_heart, on=["box_id","date"], how="left")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["box_id","date"]).reset_index(drop=True)

    # Temporal
    daily["month"]     = daily["date"].dt.month
    daily["dayofyear"] = daily["date"].dt.dayofyear
    daily["season"]    = daily["month"].map({12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3})

    # Weight features
    grp = daily.groupby("box_id")["Weight"]
    daily["weight_diff_1d"]   = grp.diff(1)
    daily["weight_diff_7d"]   = grp.diff(7)
    daily["weight_diff_14d"]  = grp.diff(14)
    daily["weight_diff_21d"]  = grp.diff(21)
    daily["weight_ma_7d"]     = grp.transform(lambda x: x.rolling(7,  min_periods=3).mean())
    daily["weight_ma_14d"]    = grp.transform(lambda x: x.rolling(14, min_periods=5).mean())
    daily["weight_std_7d"]    = grp.transform(lambda x: x.rolling(7,  min_periods=3).std())
    daily["weight_amplitude"] = daily["Weight_max"] - daily["Weight_min"]
    daily["corr_w_temp"]      = (daily.groupby("box_id", group_keys=False)
                                  .apply(lambda df: df["Weight"].rolling(14,min_periods=7)
                                         .corr(df["Temp_scale"])))

    # Acoustic
    daily["freq_ma_7d"]   = daily.groupby("box_id")["Frequency"].transform(
        lambda x: x.rolling(7, min_periods=3).mean())
    daily["freq_diff_7d"] = daily.groupby("box_id")["Frequency"].diff(7)

    # Remove known unstable periods
    _unstable = [
        (9,  "2023-08-15","2023-09-30"),
        (11, "2026-02-01","2026-04-15"),
        (17, "2025-06-01","2025-07-31"),
        (8,  "2024-09-20","2024-10-15"),
        (8,  "2024-12-15","2025-03-10"),
    ]
    mask = pd.Series(False, index=daily.index)
    for bid, s, e in _unstable:
        mask |= (daily["box_id"]==bid)&(daily["date"]>=s)&(daily["date"]<=e)
    daily = daily[~mask]

    # Clear diffs after large gaps
    daily = daily.sort_values(["box_id","date"])
    gaps  = daily.groupby("box_id")["date"].diff().dt.days > 30
    daily.loc[gaps, "weight_diff_1d"] = np.nan

    # Cyclical + extra
    daily["sin_dayofyear"]    = np.sin(2*np.pi*daily["dayofyear"]/365)
    daily["cos_dayofyear"]    = np.cos(2*np.pi*daily["dayofyear"]/365)
    spring_start              = pd.to_datetime({"year":daily["date"].dt.year,"month":3,"day":1})
    daily["days_in_season"]   = (daily["date"]-spring_start).dt.days.clip(lower=0)
    daily["weight_acceleration"] = daily.groupby("box_id")["weight_diff_1d"].diff(1)

    def _slope(arr):
        v = arr[~np.isnan(arr)]
        return float(np.polyfit(np.arange(len(v)), v, 1)[0]) if len(v) >= 5 else np.nan

    daily["weight_trend_slope"] = daily.groupby("box_id")["Weight"].transform(
        lambda x: x.rolling(14, min_periods=5).apply(_slope, raw=True))
    daily["n_positive_days_7d"] = daily.groupby("box_id")["weight_diff_1d"].transform(
        lambda x: (x>0).rolling(7, min_periods=3).sum())
    daily["temp_trend_7d"] = daily.groupby("box_id")["Temp_scale"].diff(7)

    return daily.reset_index(drop=True)


def add_hive_relative_features(daily: pd.DataFrame, df_alzas: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy().sort_values(["box_id","date"])
    for col in ["weight_historical_mean","weight_historical_std","weight_vs_historical",
                "weight_pct_of_max","days_since_last_alza","days_since_last_ADD",
                "n_alzas_this_season","weight_growing_streak"]:
        df[col] = np.nan

    for bid in df["box_id"].unique():
        mask   = df["box_id"] == bid
        df_box = df[mask].sort_values("date")
        idx    = df_box.index

        # Historical weight per month
        for month in df_box["date"].dt.month.unique():
            m_mask = df_box["date"].dt.month == month
            df_m   = df_box[m_mask].sort_values("date")
            if len(df_m) < 2: continue
            h_mean = df_m["Weight"].expanding(min_periods=7).mean().shift(1)
            h_std  = df_m["Weight"].expanding(min_periods=7).std().shift(1)
            df.loc[df_m.index,"weight_historical_mean"] = h_mean.values
            df.loc[df_m.index,"weight_historical_std"]  = h_std.values
            df.loc[df_m.index,"weight_vs_historical"]   = (
                (df_m["Weight"].values - h_mean.values) / (h_std.values + 1e-6))

        exp_max = df_box["Weight"].expanding(min_periods=14).quantile(0.95).shift(1)
        df.loc[idx,"weight_pct_of_max"] = df_box["Weight"].values / (exp_max.values + 1e-6)

        alzas_b     = df_alzas[df_alzas["box_id"]==bid].sort_values("fecha")
        alzas_b_add = alzas_b[alzas_b["accion"]=="ADD"].sort_values("fecha")

        dates_df              = df_box[["date"]].sort_values("date").reset_index()
        dates_df["date_excl"] = dates_df["date"] - pd.Timedelta(days=1)

        if len(alzas_b) > 0:
            merged = pd.merge_asof(dates_df, alzas_b[["fecha"]],
                                   left_on="date_excl", right_on="fecha", direction="backward")
            df.loc[dates_df["index"],"days_since_last_alza"] = (
                dates_df["date"] - merged["fecha"]).dt.days.fillna(999).values
        else:
            df.loc[idx,"days_since_last_alza"] = 999

        if len(alzas_b_add) > 0:
            merged_add = pd.merge_asof(dates_df, alzas_b_add[["fecha"]],
                                       left_on="date_excl", right_on="fecha", direction="backward")
            df.loc[dates_df["index"],"days_since_last_ADD"] = (
                dates_df["date"] - merged_add["fecha"]).dt.days.fillna(999).values
        else:
            df.loc[idx,"days_since_last_ADD"] = 999

        for year in df_box["date"].dt.year.unique():
            y_mask  = mask & (df["date"].dt.year == year)
            dates_y = df.loc[y_mask,"date"].sort_values()
            adds_y  = alzas_b_add[alzas_b_add["fecha"].dt.year==year]["fecha"].sort_values().values
            n = np.searchsorted(adds_y, dates_y.values, side="left") if len(adds_y) > 0 \
                else np.zeros(len(dates_y), dtype=int)
            df.loc[dates_y.index,"n_alzas_this_season"] = n

        ganando = (df_box["weight_diff_1d"] > 0).astype(int)
        streak  = ganando.groupby((ganando != ganando.shift()).cumsum()).cumsum()
        df.loc[idx,"weight_growing_streak"] = streak.values

    return df


def add_apiary_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy().sort_values(["date","box_id"])
    apiary = df.groupby("date").agg(
        apiary_weight_mean=("Weight","mean"),
        apiary_weight_std =("Weight","std"),
        apiary_growth_mean=("weight_diff_7d","mean"),
        apiary_growth_std =("weight_diff_7d","std"),
        n_hives_active    =("Weight","count"),
    ).reset_index()
    df = df.merge(apiary, on="date", how="left")
    df["weight_vs_apiary"]   = (df["Weight"]-df["apiary_weight_mean"])/(df["apiary_weight_std"]+1e-6)
    df["growth_vs_apiary"]   = (df["weight_diff_7d"]-df["apiary_growth_mean"])/(df["apiary_growth_std"]+1e-6)
    df["weight_rank_pct"]    = df.groupby("date")["Weight"].rank(pct=True)
    df["growth_rank_pct"]    = df.groupby("date")["weight_diff_7d"].rank(pct=True)
    df["weight_apiary_ratio"]= df["Weight"]/(df["apiary_weight_mean"]+1e-6)
    return df


def add_threshold_features(daily: pd.DataFrame) -> pd.DataFrame:
    df  = daily.copy()
    act = [3,4,5,6,7]
    df["above_min_weight"] = ((df["Weight"]>=20)&(df["month"].isin(act))).astype(int)
    df["overdue_for_alza"] = ((df["days_since_last_alza"]>200)&(df["month"].isin(act))&(df["Weight"]>=20)).astype(int)
    df["small_but_growing"]= ((df["weight_rank_pct"]<0.5)&(df["weight_diff_14d"]>1.0)&(df["month"].isin(act))).astype(int)
    return df


def run_feature_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    df_alzas = _build_df_alzas()

    print("[alza] Building daily features ...", file=sys.stderr)
    daily = build_daily_features(raw_df)
    print(f"[alza]   daily: {len(daily):,} rows", file=sys.stderr)

    # Remove 2025 harvest period + empty hives
    daily = daily[~((daily["date"]>="2025-05-12")&(daily["date"]<="2025-05-15"))]
    vacias_2025 = [10,11,12,16,17,18]
    daily = daily[~((daily["date"].dt.year==2025)&(daily["box_id"].isin(vacias_2025)))]

    print("[alza] Adding hive-relative features ...", file=sys.stderr)
    daily = add_hive_relative_features(daily, df_alzas)

    print("[alza] Adding apiary-comparison features ...", file=sys.stderr)
    daily = add_apiary_features(daily)

    print("[alza] Adding threshold features ...", file=sys.stderr)
    daily = add_threshold_features(daily)

    # Filter to active season (Feb–Jun) — same as training
    daily = daily[daily["month"].between(2, 6)]

    print(f"[alza] Final: {len(daily):,} rows | "
          f"{daily['date'].min().date()} → {daily['date'].max().date()}", file=sys.stderr)
    return daily.reset_index(drop=True)


# ── Main ───────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",        type=int,  default=30,
                        help="Days of history to download (default 30)")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip download, rebuild features from existing raw CSV")
    args = parser.parse_args()

    t0        = datetime.now()
    new_rows  = 0
    error_msg = None

    try:
        if args.no_download:
            print("[alza] Skipping download (--no-download)", file=sys.stderr)
            if not RAW_CSV.exists():
                raise FileNotFoundError(f"Raw CSV not found: {RAW_CSV}")
            raw_df = pd.read_csv(str(RAW_CSV), parse_dates=["Time"], low_memory=False)
            if "Hive name" not in raw_df.columns and "box_id" in raw_df.columns:
                raw_df = raw_df.rename(columns={"box_id":"Hive name"})
        else:
            xlsx_bytes = await download_excel(days=args.days)
            # Save to disk for debugging / recovery
            _debug_xlsx = HERE / "debug_last_download.xlsx"
            _debug_xlsx.write_bytes(xlsx_bytes)
            print(f"[alza] xlsx saved to {_debug_xlsx}", file=sys.stderr)
            new_df     = xlsx_to_raw_df(xlsx_bytes)
            new_rows   = len(new_df)
            raw_df     = merge_raw(new_df)
            save_raw(raw_df)

        # Rebuild features
        feat_df = run_feature_pipeline(raw_df)

        # Save to both locations
        feat_df.to_csv(str(FEAT_CSV), index=False)
        FEAT_CSV2.parent.mkdir(parents=True, exist_ok=True)
        feat_df.to_csv(str(FEAT_CSV2), index=False)
        print(f"[alza] Saved: {FEAT_CSV}", file=sys.stderr)
        print(f"[alza] Saved: {FEAT_CSV2}", file=sys.stderr)

        elapsed = (datetime.now() - t0).seconds
        last_date_per_hive = (feat_df.groupby("box_id")["date"]
                               .max().apply(lambda x: str(x.date())).to_dict())
        output = {
            "status":       "ok",
            "run_date":     datetime.today().strftime("%Y-%m-%d"),
            "elapsed_s":    elapsed,
            "new_raw_rows": new_rows,
            "total_feat_rows": len(feat_df),
            "hives":        sorted(feat_df["box_id"].unique().tolist()),
            "last_date_per_hive": {str(k): v for k,v in last_date_per_hive.items()},
            "feat_csv":     str(FEAT_CSV),
        }

    except Exception as exc:
        import traceback
        error_msg = str(exc)
        print(f"[alza] ERROR: {error_msg}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        output = {
            "status":    "error",
            "run_date":  datetime.today().strftime("%Y-%m-%d"),
            "error":     error_msg,
        }

    print(json.dumps(output, ensure_ascii=False))
    return 0 if output["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
