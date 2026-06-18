"""
Weekly swarm-risk pipeline:
  1. Opens beehivemonitoring.com with Playwright (share URL auto-auth)
  2. Clicks Menu -> Export Excel, selects all 8 swarm hives, sets last 60 days
  3. Downloads the xlsx (one file, all hives at once)
  4. Reads it and calls FastAPI /predict for each hive
  5. Prints JSON summary to stdout (n8n reads it)

Install once:
    pip install playwright openpyxl requests
    python -m playwright install chromium

Run:
    python data_fetcher.py [--days 60] [--api http://localhost:8000]
"""

import argparse
import asyncio
import io
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import openpyxl
import pandas as pd
import requests
from playwright.async_api import async_playwright

SHARE_URL = "https://main.beehivemonitoring.com/c36f58c6b327462fa1b23da7f652697d"

# Hive name prefixes to select (matching the checkbox labels)
SWARM_HIVE_PREFIXES = ["001", "002", "003", "004", "005", "008", "013", "014"]

# box_id numbers the model uses (int version of the prefix)
SWARM_BOX_IDS = [1, 2, 3, 4, 5, 8, 13, 14]


def _fmt_date(dt: datetime) -> str:
    """M/D/YYYY without leading zeros (matches the dialog format)."""
    return f"{dt.month}/{dt.day}/{dt.year}"


async def download_excel(days: int = 60) -> bytes:
    today = datetime.today()
    date_from = _fmt_date(today - timedelta(days=days))
    date_to = _fmt_date(today)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(accept_downloads=True)
        page = await ctx.new_page()

        print(f"[fetcher] Abriendo {SHARE_URL} ...", file=sys.stderr)
        await page.goto(SHARE_URL, wait_until="networkidle", timeout=90_000)

        # --- Open hamburger menu (top-right ☰) ---
        # The button contains a Material icon with text "menu"
        await page.locator('button').filter(has_text="menu").last.click()
        await page.wait_for_selector("text=Export Excel", timeout=8_000)
        await page.get_by_text("Export Excel").click()
        await page.wait_for_selector("text=Date from", timeout=8_000)

        # --- Set "Date from" (editable text field) ---
        date_from_input = page.locator('input').first
        await date_from_input.triple_click()
        await date_from_input.fill(date_from)
        await page.keyboard.press("Tab")

        # --- Set "Date to" ---
        date_to_input = page.locator('input').nth(1)
        await date_to_input.triple_click()
        await date_to_input.fill(date_to)
        await page.keyboard.press("Tab")

        print(f"[fetcher] Rango: {date_from} → {date_to}", file=sys.stderr)

        # --- Uncheck all, then check only swarm hives ---
        all_checkboxes = page.locator('mat-checkbox, input[type="checkbox"]')
        n = await all_checkboxes.count()
        for i in range(n):
            cb = all_checkboxes.nth(i)
            # mat-checkbox: check via the inner input
            inner = cb.locator('input[type="checkbox"]')
            if await inner.count() > 0:
                if await inner.is_checked():
                    await inner.click(force=True)
            else:
                if await cb.is_checked():
                    await cb.click(force=True)

        checked = 0
        for prefix in SWARM_HIVE_PREFIXES:
            # Find the checkbox whose label contains the prefix
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
                print(f"  [?] {prefix} no encontrada", file=sys.stderr)

        print(f"[fetcher] {checked}/{len(SWARM_HIVE_PREFIXES)} colmenas seleccionadas", file=sys.stderr)

        # --- Download excel ---
        async with page.expect_download(timeout=60_000) as dl_info:
            await page.get_by_role("button", name="Download excel").click()
        download = await dl_info.value

        path = await download.path()
        data = Path(path).read_bytes()
        print(f"[fetcher] Descargado: {len(data):,} bytes", file=sys.stderr)

        await ctx.close()
        await browser.close()
        return data


def xlsx_to_df(xlsx_bytes: bytes) -> pd.DataFrame:
    """Parse the exported xlsx into a DataFrame matching the model's expected format."""
    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes), read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    if not rows:
        raise ValueError("El xlsx está vacío")

    headers = [str(h) if h is not None else "" for h in rows[0]]
    df = pd.DataFrame(rows[1:], columns=headers)

    # The xlsx already has the exact column names the model expects
    needed = ["Hive name", "Time", "Weight", "Frequency", "Volume",
              "Temperature heart", "Humidity heart", "Temperature scale", "Humidity scale"]
    df = df[[c for c in needed if c in df.columns]].copy()

    # Time strings have NARROW NO-BREAK SPACE before AM/PM: "6/1/2026 10:00:21 AM"
    # Strip it and use a fixed format for fast C-level parsing
    df["Time"] = pd.to_datetime(
        df["Time"].astype(str).str.replace(" ", " ", regex=False),
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )
    # "001 (I*) Blanca" → 1
    df["Hive name"] = (
        df["Hive name"].astype(str).str.extract(r"^0*(\d+)")[0].astype(int)
    )
    df = df.dropna(subset=["Time"]).sort_values(["Hive name", "Time"]).reset_index(drop=True)
    # Cap to last 90 days so data size stays bounded even if the export is large
    cutoff = df["Time"].max() - pd.Timedelta(days=90)
    df = df[df["Time"] >= cutoff].reset_index(drop=True)
    return df


def predict_hive(api_url: str, df: pd.DataFrame, box_id: int) -> dict:
    hive_df = df[df["Hive name"] == box_id].copy()
    if len(hive_df) < 14:
        return {"box_id": box_id, "risk_level": "SIN_DATOS",
                "error": f"Solo {len(hive_df)} filas (se necesitan >=14)"}
    # Restore original "Hive name" string expected by inference.py
    hive_df["Hive name"] = box_id
    csv_bytes = hive_df.to_csv(index=False).encode()
    try:
        r = requests.post(
            f"{api_url}/predict?box_id={box_id}",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        return {"box_id": box_id, "risk_level": "ERROR",
                "error": f"HTTP {r.status_code}: {r.text[:300]}"}
    except requests.exceptions.ConnectionError:
        return {"box_id": box_id, "risk_level": "ERROR",
                "error": "No se puede conectar con la API (¿está corriendo uvicorn?)"}
    except Exception as e:
        return {"box_id": box_id, "risk_level": "ERROR", "error": str(e)}


async def main():
    parser = argparse.ArgumentParser(description="Weekly swarm risk pipeline")
    parser.add_argument("--days", type=int, default=60,
                        help="Dias de historial a descargar (default 60)")
    parser.add_argument("--api", default="http://localhost:8000",
                        help="URL de la FastAPI (default http://localhost:8000)")
    parser.add_argument("--save-xlsx", default=None,
                        help="Ruta opcional para guardar el xlsx descargado")
    args = parser.parse_args()

    # 1. Download
    xlsx_bytes = await download_excel(days=args.days)
    if args.save_xlsx:
        Path(args.save_xlsx).write_bytes(xlsx_bytes)

    # 2. Parse
    df = xlsx_to_df(xlsx_bytes)
    hives_found = sorted(df["Hive name"].unique().tolist())
    print(f"[fetcher] {len(df):,} filas | colmenas: {hives_found}", file=sys.stderr)

    # 3. Predict per hive
    results = []
    for box_id in hives_found:
        res = predict_hive(args.api, df, box_id)
        results.append(res)
        pct = f"{res['swarm_risk_probability']*100:.1f}%" if "swarm_risk_probability" in res else res.get("error", "?")
        print(f"  Hive {box_id:>2}: {res.get('risk_level','?'):10} {pct}", file=sys.stderr)

    # 4. JSON to stdout (n8n reads this)
    output = {
        "status": "ok",
        "run_date": datetime.today().strftime("%Y-%m-%d"),
        "hives": hives_found,
        "predictions": results,
        "alerts": [r for r in results if r.get("risk_level") == "ALTO"],
        "alert_count": sum(1 for r in results if r.get("risk_level") == "ALTO"),
    }
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
