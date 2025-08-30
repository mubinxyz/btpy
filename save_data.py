import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# --- improved helper to convert many input types to unix seconds ---
def to_unix_timestamp(value, assume_tz: str | None = "Asia/Qatar") -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        iv = int(value)
        if iv > 1_000_000_000_000:
            return iv // 1000
        return iv
    if isinstance(value, (pd.Timestamp, datetime)):
        ts = pd.to_datetime(value)
        if ts.tz is None:
            ts = ts.tz_localize(assume_tz if assume_tz else "UTC")
        ts = ts.tz_convert("UTC")
        return int(ts.timestamp())
    if isinstance(value, str):
        ts = pd.to_datetime(value, errors="raise")
        if ts.tz is None:
            ts = ts.tz_localize(assume_tz if assume_tz else "UTC")
        ts = ts.tz_convert("UTC")
        return int(ts.timestamp())
    raise TypeError(f"Unsupported type for timestamp conversion: {type(value)}")


# --- normalize OHLC and convert datetimes ---
def normalize_ohlc(ohlc_data: dict, return_tz_offset_minutes: int = 210) -> pd.DataFrame:
    if not ohlc_data:
        return pd.DataFrame()

    t_list = list(ohlc_data.get("t", []))
    max_val = 0
    for x in t_list:
        try:
            if x is not None and pd.notna(x):
                max_val = max(max_val, int(x))
        except Exception:
            pass
    if max_val > 1_000_000_000_000:  
        t_list = [int(x) // 1000 if (x is not None and pd.notna(x)) else None for x in t_list]

    target_tz = timezone(timedelta(minutes=return_tz_offset_minutes))  
    datetimes = []
    for ts in t_list:
        if ts is None or pd.isna(ts):
            datetimes.append(pd.NaT)
            continue
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(target_tz)
        datetimes.append(dt)

    df = pd.DataFrame({
        "Date": pd.DatetimeIndex(datetimes),
        "Open": pd.to_numeric(ohlc_data.get("o", []), errors="coerce"),
        "High": pd.to_numeric(ohlc_data.get("h", []), errors="coerce"),
        "Low": pd.to_numeric(ohlc_data.get("l", []), errors="coerce"),
        "Close": pd.to_numeric(ohlc_data.get("c", []), errors="coerce"),
    })

    if "v" in ohlc_data:
        df["Volume"] = pd.to_numeric(ohlc_data.get("v", []), errors="coerce")
    else:
        df["Volume"] = 0

    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ensure backtesting.py compatibility
    df["Date"] = df["Date"].dt.tz_localize(None)  # remove timezone
    return df


# --- get_ohlc with input timezone option ---
def get_ohlc(symbol: str, timeframe: str,
             from_date=None, to_date=None, input_tz: str | None = "Asia/Qatar") -> pd.DataFrame:
    if to_date is None:
        to_date = time.time()
    if from_date is None:
        from_date = to_date - 86400

    from_unix = to_unix_timestamp(from_date, assume_tz=input_tz)
    to_unix   = to_unix_timestamp(to_date,   assume_tz=input_tz)
    if from_unix is None or to_unix is None:
        raise ValueError("from_date and to_date must be convertable to unix seconds")

    try:
        lite_finance_url = (
            "https://my.litefinance.org/chart/get-history"
            f"?symbol={symbol.upper()}&resolution={timeframe}&from={from_unix}&to={to_unix}"
        )
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Referer": "https://my.litefinance.org/",
            "X-Requested-With": "XMLHttpRequest",
        }
        resp = requests.get(lite_finance_url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        ohlc_data = data.get("data", {})
        if ohlc_data:
            return normalize_ohlc(ohlc_data, return_tz_offset_minutes=3 * 60 + 30)
    except Exception as e:
        print(f"[LiteFinance] OHLC error: {e}")
    return pd.DataFrame()


# --- save df to csv for backtesting.py ---
def save_to_csv(df: pd.DataFrame, symbol: str, timeframe: str):
    os.makedirs("csv_data", exist_ok=True)
    filename = f"csv_data/csv_{symbol.lower()}_{timeframe}.csv"
    # ensure correct CSV format
    df.to_csv(filename, index=False, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    print(f"Saved: {filename}")


# --- Example usage ---
if __name__ == "__main__":
    df = get_ohlc("btcusd", "5",
                  
                  input_tz="Asia/Qatar")
    if not df.empty:
        save_to_csv(df, "btcusd", "5")
        print(df.head())
