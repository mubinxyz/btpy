# ohlc_scheduler.py
"""
OHLC scheduler with timezone normalization fix (ensures all datetimes are tz-aware in USER_TZ).
Keeps chunking, timeouts, retries, file-locking, atomic writes, initial full sync.
"""
import os
import time
import random
import logging
import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout

import pandas as pd

# Import your provider function (adjust if signature differs)
from save_data import get_ohlc

# ----------------- CONFIG -----------------
USER_TZ = "Asia/Qatar"        # timezone used for aligning schedule and passing to get_ohlc
TZ = ZoneInfo(USER_TZ)
N_CANDLES = 10000            # number of candles desired overall per timeframe
CHUNK_CANDLES = 1000         # max candles per API call during chunked historic fetch
MAX_WORKERS = 6              # concurrency for scheduled fetches
MAX_WORKERS_INIT = 3         # concurrency for initial full sync
CALL_TIMEOUT = 40            # seconds per get_ohlc call
RETRY_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 4
JITTER_MAX = 1.5
RATE_LIMIT_SLEEP_ON_429 = 60
INTER_CHUNK_SLEEP = 0.5
DATA_DIR = "csv_data"
LOG_FILE = "ohlc_scheduler.log"
# ------------------------------------------

# Logging
os.makedirs(DATA_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger("ohlc_sched")

# Symbols & timeframes (adjust your provider names if needed)
CRYPTO = ["btcusd", "ethusd", "xrpusd"]
FOREX = ["usdx", "eurusd", "gbpusd", "chfusd", "audusd", "nzdusd", "usdcad"]
SYMBOLS = CRYPTO + FOREX

TIMEFRAMES = ["1", "5", "15", "30", "60", "240", "D"]
TF_SECONDS = {"1":60,"5":5*60,"15":15*60,"30":30*60,"60":3600,"240":4*3600,"D":24*3600}

# file locks
_file_locks = {}
_file_locks_lock = threading.Lock()
def _get_file_lock(filename: str):
    with _file_locks_lock:
        lk = _file_locks.get(filename)
        if lk is None:
            lk = threading.Lock()
            _file_locks[filename] = lk
        return lk

# ---------- Timezone helpers ----------
def ensure_tz_ts(ts, tz=TZ):
    """
    Ensure ts (datetime / pandas.Timestamp / str / numpy datetime) is a pandas.Timestamp
    that is timezone-aware in tz. Returns pandas.Timestamp or pd.NaT.
    """
    if ts is None:
        return pd.NaT
    # Convert to pandas.Timestamp first
    try:
        ts = pd.Timestamp(ts)
    except Exception:
        return pd.NaT
    if pd.isna(ts):
        return pd.NaT
    # If tz-naive -> localize to tz
    if ts.tzinfo is None or getattr(ts, "tz", None) is None:
        try:
            # tz_localize works on pandas.Timestamp
            return ts.tz_localize(tz)
        except Exception:
            # fallback: set tzinfo by replace (less correct for ambiguous times, but ok)
            return ts.replace(tzinfo=tz)
    else:
        # convert to target tz
        try:
            return ts.tz_convert(tz)
        except Exception:
            return ts.astimezone(tz)

# ---------- DF normalization ----------
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure DataFrame has Date column as timezone-aware pandas.Timestamp in USER_TZ,
       and the columns: Date, Open, High, Low, Close, Volume (create missing columns).
    """
    if df is None:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])

    # If Date is index, move it to column
    if "Date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.index.name or "index": "Date"})

    # try to find a date-like column if Date missing
    if "Date" not in df.columns:
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                df = df.rename(columns={col: "Date"})
                break

    # Ensure Date exists
    if "Date" not in df.columns:
        df["Date"] = pd.NaT

    # Parse and enforce timezone for Date column
    # Use apply to ensure each entry becomes tz-aware
    df["Date"] = df["Date"].apply(lambda x: ensure_tz_ts(x, TZ))

    # Ensure other columns exist
    want = ["Date","Open","High","Low","Close","Volume"]
    for c in want:
        if c not in df.columns:
            df[c] = pd.NA

    # Drop rows where Date is NaT
    df = df[want].copy()
    df = df[df["Date"].notna()]
    # sort by Date ascending
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ---------- safe get_ohlc with timeout + retries ----------
def safe_get_ohlc_call(symbol: str, timeframe: str, from_date, to_date, input_tz=USER_TZ, **kwargs):
    attempt = 0
    while attempt < RETRY_ATTEMPTS:
        attempt += 1
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(get_ohlc, symbol, timeframe, from_date, to_date, input_tz, **kwargs)
                df = fut.result(timeout=CALL_TIMEOUT)
            if df is None:
                return pd.DataFrame()
            df = normalize_df(df)
            return df
        except FutureTimeout:
            wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, JITTER_MAX)
            logger.warning("[%s %s] call TIMEOUT after %ss (attempt %d). Backoff %.1fs",
                           symbol, timeframe, CALL_TIMEOUT, attempt, wait)
            time.sleep(wait)
            continue
        except Exception as e:
            s = str(e).lower()
            is_429 = False
            try:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 429:
                    is_429 = True
            except Exception:
                pass
            if "429" in s or "rate limit" in s or is_429:
                wait = RATE_LIMIT_SLEEP_ON_429 * attempt
                logger.warning("[%s %s] detected rate-limit/429: %s -- sleeping %ds", symbol, timeframe, repr(e), wait)
                time.sleep(wait)
                continue
            else:
                wait = RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, JITTER_MAX)
                logger.warning("[%s %s] call error: %s (attempt %d). Backoff %.1fs", symbol, timeframe, repr(e), attempt, wait)
                time.sleep(wait)
                continue
    logger.error("[%s %s] all retries failed.", symbol, timeframe)
    return pd.DataFrame()

# ---------- chunked historic fetch (with timezone-safe arithmetic) ----------
def fetch_in_chunks(symbol: str, timeframe: str, from_date, to_date, chunk_candles: int = CHUNK_CANDLES, incremental: bool = False):
    """
    Fetch data in chunks. from_date and to_date can be naive/aware; they will be normalized to USER_TZ.
    Returns combined df or empty df.
    """
    # normalize endpoints to tz-aware pandas.Timestamp
    from_date = ensure_tz_ts(from_date, TZ)
    to_date = ensure_tz_ts(to_date, TZ)
    if pd.isna(from_date) or pd.isna(to_date) or from_date >= to_date:
        logger.info("[%s %s] invalid or empty range after tz-normalize: %s -> %s", symbol, timeframe, from_date, to_date)
        return pd.DataFrame()

    seconds_per_candle = TF_SECONDS.get(timeframe, 60)
    chunk_seconds = seconds_per_candle * chunk_candles

    current_start = from_date
    parts = []
    max_chunks = max(1, (N_CANDLES // max(1, chunk_candles)) + 5)
    chunk_count = 0

    while current_start < to_date and chunk_count < max_chunks:
        chunk_count += 1
        current_end = current_start + timedelta(seconds=chunk_seconds)
        if current_end > to_date:
            current_end = to_date
        logger.info("[%s %s] fetching chunk %s -> %s", symbol, timeframe, current_start.isoformat(), current_end.isoformat())
        df_chunk = safe_get_ohlc_call(symbol, timeframe, current_start, current_end)
        if df_chunk is None or df_chunk.empty:
            logger.info("[%s %s] chunk returned empty (end %s). Stopping chunk loop.", symbol, timeframe, current_end.isoformat())
            break
        # ensure df_chunk Dates are in USER_TZ (normalize_df already did)
        parts.append(df_chunk)
        # move start forward to last timestamp + one candle
        try:
            last_ts = parts[-1]["Date"].max()
            last_ts = ensure_tz_ts(last_ts, TZ)
            current_start = last_ts + timedelta(seconds=seconds_per_candle)
        except Exception:
            current_start = current_end + timedelta(seconds=1)
        time.sleep(INTER_CHUNK_SLEEP)

    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    combined.drop_duplicates(subset=["Date"], inplace=True)
    combined.sort_values("Date", inplace=True)
    # limit to requested range and N_CANDLES
    combined = combined[(combined["Date"] >= from_date) & (combined["Date"] <= to_date)]
    if len(combined) > N_CANDLES:
        combined = combined.iloc[-N_CANDLES:].reset_index(drop=True)
    combined.reset_index(drop=True, inplace=True)
    return combined

# ---------- atomic save with per-file lock (ensuring Dates tz-aware) ----------
def save_df_atomic(df: pd.DataFrame, filename: str):
    lock = _get_file_lock(filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with lock:
        # read existing
        if os.path.exists(filename):
            try:
                existing = pd.read_csv(filename, parse_dates=["Date"])
                # convert existing Date to tz-aware
                if not existing.empty:
                    existing["Date"] = existing["Date"].apply(lambda x: ensure_tz_ts(x, TZ))
            except Exception:
                existing = pd.DataFrame()
        else:
            existing = pd.DataFrame()

        # ensure new df Date tz-aware
        df["Date"] = df["Date"].apply(lambda x: ensure_tz_ts(x, TZ))
        if not existing.empty:
            combined = pd.concat([existing, df], ignore_index=True)
            combined.drop_duplicates(subset=["Date"], inplace=True)
            combined.sort_values("Date", inplace=True)
        else:
            combined = df.copy()
            combined.sort_values("Date", inplace=True)

        want = ["Date","Open","High","Low","Close","Volume"]
        for c in want:
            if c not in combined.columns:
                combined[c] = pd.NA
        combined = combined[want]
        tmp = filename + ".tmp"
        combined.to_csv(tmp, index=False)
        os.replace(tmp, filename)
        logger.info("Saved %d rows to %s (total %d)", len(df), filename, len(combined))
    return True

def last_saved_timestamp(symbol: str, timeframe: str):
    filename = os.path.join(DATA_DIR, f"csv_{symbol.lower()}_{timeframe}.csv")
    if not os.path.exists(filename):
        return None
    try:
        df = pd.read_csv(filename, parse_dates=["Date"])
        if df.empty:
            return None
        max_ts = pd.to_datetime(df["Date"].max())
        return ensure_tz_ts(max_ts, TZ)
    except Exception:
        return None

# ---------- high-level fetch & store ----------
def fetch_and_store(symbol: str, timeframe: str, now_local: datetime, n_candles=N_CANDLES, incremental=True):
    try:
        # normalize now_local
        now_local = ensure_tz_ts(now_local, TZ)
        seconds = TF_SECONDS.get(timeframe, 60)
        if incremental:
            last_ts = last_saved_timestamp(symbol, timeframe)
            if last_ts is not None:
                from_date = last_ts + timedelta(seconds=seconds)
            else:
                from_date = now_local - timedelta(seconds=seconds * n_candles)
        else:
            from_date = now_local - timedelta(seconds=seconds * n_candles)
        to_date = now_local
        if from_date >= to_date:
            logger.info("[%s %s] no range to fetch (from >= to).", symbol, timeframe)
            return False

        df = fetch_in_chunks(symbol, timeframe, from_date, to_date, chunk_candles=CHUNK_CANDLES, incremental=incremental)
        if df is None or df.empty:
            logger.info("[%s %s] no new data returned.", symbol, timeframe)
            return False

        filename = os.path.join(DATA_DIR, f"csv_{symbol.lower()}_{timeframe}.csv")
        save_df_atomic(df, filename)
        return True
    except Exception as e:
        logger.exception("fetch_and_store exception for %s %s: %s", symbol, timeframe, e)
        return False

# ---------- initial full sync ----------
def initial_full_sync(symbols, timeframes):
    logger.info("Starting initial full sync: %d symbols x %d timeframes (up to %d candles each)",
                len(symbols), len(timeframes), N_CANDLES)
    now_local = ensure_tz_ts(datetime.now(tz=TZ), TZ)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_INIT) as ex:
        futures = []
        for tf in timeframes:
            for sym in symbols:
                futures.append(ex.submit(fetch_and_store, sym, tf, now_local, N_CANDLES, False))
        for f in as_completed(futures):
            try:
                _ = f.result()
            except Exception as e:
                logger.exception("Initial sync worker exception: %s", e)
    logger.info("Initial full sync finished.")

# ---------- scheduler ----------
def timeframe_should_run(now: datetime, tf: str) -> bool:
    if now.second != 0:
        return False
    m = now.minute
    h = now.hour
    if tf == "1":
        return True
    if tf == "5":
        return (m % 5) == 0
    if tf == "15":
        return (m % 15) == 0
    if tf == "30":
        return (m % 30) == 0
    if tf == "60":
        return m == 0
    if tf == "240":
        return (m == 0) and ((h % 4) == 0)
    if tf == "D":
        return (h == 0 and m == 0)
    return False

def align_to_next_minute(tz_name: str):
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz=tz)
    next_minute = (now.replace(second=0, microsecond=0) + timedelta(minutes=1))
    wait = (next_minute - now).total_seconds()
    if wait > 0:
        time.sleep(wait)

def scheduler_loop():
    logger.info("Starting scheduler loop (timezone=%s). Aligning to next minute...", USER_TZ)
    align_to_next_minute(USER_TZ)
    try:
        while True:
            now_local = ensure_tz_ts(datetime.now(tz=TZ), TZ).replace(microsecond=0)
            due_tfs = [tf for tf in TIMEFRAMES if timeframe_should_run(now_local, tf)]
            if due_tfs:
                logger.info("[%s] Due timeframes: %s", now_local.isoformat(), due_tfs)
                futures = []
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                    for tf in due_tfs:
                        for sym in SYMBOLS:
                            futures.append(ex.submit(fetch_and_store, sym, tf, now_local, N_CANDLES, True))
                    for f in as_completed(futures):
                        try:
                            _ = f.result()
                        except Exception as e:
                            logger.exception("Scheduled worker error: %s", e)
            align_to_next_minute(USER_TZ)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user (KeyboardInterrupt).")

# ---------- main ----------
if __name__ == "__main__":
    logger.info("OHLC Scheduler starting up (timezone fix applied).")
    try:
        initial_full_sync(SYMBOLS, TIMEFRAMES)
    except Exception:
        logger.exception("Initial full sync failed unexpectedly.")
    scheduler_loop()
