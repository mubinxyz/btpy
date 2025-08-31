# strategy/smc_test.py

from backtesting import Strategy, Backtest
import pandas as pd
import pandas_ta as ta
import numpy as np
from backtesting.lib import plot_heatmaps, crossover, barssince, FractionalBacktest, resample_apply
from smartmoneyconcepts import smc
import mplfinance as mpf # for img output


csv_data_name = "btcusd_5"
df = pd.read_csv(f'../csv_data/csv_{csv_data_name}.csv', parse_dates=True, index_col=[0])[:500]

# print(df.head())
df.info()



#  --- MA Indicators ---
def atr(data, atr_length):
    atr = ta.atr(high=data.High, low=data.Low, close=data.Close, length=atr_length)
    return atr.to_numpy().T

def ssma(close_data, ssma_length) -> pd.Series:
    ssma = close_data.copy()
    ssma.iloc[:ssma_length] = close_data.iloc[:ssma_length].mean()  # initialize with SMA
    for i in range(ssma_length, len(close_data)):
        ssma.iloc[i] = (ssma.iloc[i - 1] * (ssma_length - 1) + close_data.iloc[i]) / ssma_length
    return ssma

def ema(close_data, ema_length):
    ema = ta.ema(close=close_data, length=ema_length)
    return ema.to_numpy().T

# --- Helper to build OHLCV DataFrame ---
def _prepare_ohlcv(data):
    return pd.DataFrame({
        'open':   data.Open.s,
        'high':   data.High.s,
        'low':    data.Low.s,
        'close':  data.Close.s,
        'volume': data.Volume.s
    }, index=data.index)

# --- Normalizer: convert swings output to consistent formats ----------

def _swings_to_transposed_array(sw):
    """
    Return a transposed numpy array where axis 0 = columns of original swings (so
    index by [col_idx][-1] in your strategy). If cannot determine, returns
    an array of NaNs with shape (2, n).
    """
    # If DataFrame -> to_numpy then transpose
    if isinstance(sw, pd.DataFrame):
        try:
            arr = sw.to_numpy().T
            return arr
        except Exception:
            pass

    # If ndarray -> try to interpret shapes
    if isinstance(sw, np.ndarray):
        # If already shape (n_cols, n_rows) -> assume it's already transposed
        if sw.ndim == 2:
            # if first dim matches a small number of "columns" (like 2 or 3),
            # and second dim == length, it's already in column-first layout.
            if sw.shape[0] <= 10 and sw.shape[1] > sw.shape[0]:
                # assume (n_cols, n_rows)
                return sw
            # else if rows align to bars (n_rows, n_cols), transpose to (n_cols, n_rows)
            if sw.shape[0] > sw.shape[1]:
                return sw.T
            # else default to transpose as a best-effort
            return sw.T
        # 1D -> produce 2 x N with first row = data, second row = NaNs
        if sw.ndim == 1:
            n = sw.shape[0]
            arr = np.vstack([sw, np.full(n, np.nan)])
            return arr

    # Fallback -> return NaN array with 2 rows same length as df (unknown length: choose 1)
    return np.array([[np.nan], [np.nan]])

# --- Named wrappers for SMC indicators ---

def fvg_signal(data, join_consecutive=False):
    """1 if bullish FVG, -1 if bearish, 0 none"""
    ohlc = _prepare_ohlcv(data)
    df = smc.fvg(ohlc, join_consecutive=join_consecutive)
    return df.to_numpy().T


def swing_highs_lows_series(data, swing_length=50):
    """Returns swings as transposed NumPy array: rows -> columns of original DataFrame or array"""
    ohlc = _prepare_ohlcv(data)
    sw = smc.swing_highs_lows(ohlc, swing_length=swing_length)
    arr = _swings_to_transposed_array(sw)
    return arr


def bos_signal(data, swing_length=50, close_break=True):
    """1 bullish BOS, -1 bearish, 0 none"""
    ohlc = _prepare_ohlcv(data)
    swings = smc.swing_highs_lows(ohlc, swing_length=swing_length)  # passed to bos_choch
    df = smc.bos_choch(ohlc, swings, close_break=close_break)
    return df.to_numpy().T


def ob_signal(data, swing_length=50, close_mitigation=False):
    """1 bullish OB, -1 bearish, 0 none"""
    ohlc = _prepare_ohlcv(data)
    swings = smc.swing_highs_lows(ohlc, swing_length=swing_length)
    df = smc.ob(ohlc, swings, close_mitigation=close_mitigation)
    return df.to_numpy().T


def liquidity_signal(data, swing_length=50, range_percent=0.01):
    """1 liquidity present, else 0"""
    ohlc = _prepare_ohlcv(data)
    swings = smc.swing_highs_lows(ohlc, swing_length=swing_length)
    df = smc.liquidity(ohlc, swings, range_percent=range_percent)
    return df.to_numpy().T


def prev_high(data, time_frame="1D"):
    """Previous period high"""
    ohlc = _prepare_ohlcv(data)
    df = smc.previous_high_low(ohlc, time_frame=time_frame)
    return df['PreviousHigh'].to_numpy().T

def prev_low(data, time_frame="1D"):
    """Previous period low"""
    ohlc = _prepare_ohlcv(data)
    df = smc.previous_high_low(ohlc, time_frame=time_frame)
    return df['PreviousLow'].to_numpy().T


def session_open(data, session, start_time=None, end_time=None, time_zone="UTC"):
    """1 if in session, else 0"""
    ohlc = _prepare_ohlcv(data)
    df = smc.sessions(ohlc, session=session, start_time=start_time, end_time=end_time, time_zone=time_zone)
    return df.to_numpy().T


def retrace_level(data, swing_length=50):
    """Retracement level"""
    ohlc = _prepare_ohlcv(data)
    swings = smc.swing_highs_lows(ohlc, swing_length=swing_length)
    df = smc.retracements(ohlc, swings)
    return df.to_numpy().T

# ATR as NumPy array
def atr_array(data, atr_period):
    series = ta.atr(high=data.High.s, low=data.Low.s, close=data.Close.s, length=atr_period)
    return series.to_numpy().T

# --- Strategy using full set of wrappers ---
class SMC(Strategy):
    atr_period = 14
    atr_coe = 2
    rrr = 2
    swing_length = 10
    bos_close_break = True
    ob_mitigation = False
    liq_range = 0.01
    prev_timeframe = "1D"
    session_name = "New York"
    session_tz = "UTC"

    def init(self):

        # self.atr = self.I(atr_array, self.data, self.atr_period)
        self.fvg = self.I(fvg_signal, self.data, plot=False)
        self.swing_hl = self.I(swing_highs_lows_series, self.data, swing_length=self.swing_length, plot=True)
        self.bos = self.I(bos_signal, self.data, swing_length=self.swing_length, close_break=self.bos_close_break, plot=False)
        self.ob  = self.I(ob_signal, self.data, swing_length=self.swing_length, close_mitigation=self.ob_mitigation, plot=False)
        # self.liq = self.I(liquidity_signal, self.data, swing_length=self.swing_length, range_percent=self.liq_range, plot=False)
        # self.prev_h = self.I(prev_high, self.data, time_frame=self.prev_timeframe, plot=False)
        # self.prev_l = self.I(prev_low, self.data, time_frame=self.prev_timeframe, plot=False)
        # self.sess = self.I(session_open, self.data, session=self.session_name, time_zone=self.session_tz)
        # self.retr = self.I(retrace_level, self.data, swing_length=self.swing_length, plot=False)

        self.sl_price = None
        self.tp_price = None

    def next(self):
        # Accessing the transposed numpy: [column_index][-1]
        # column 0 -> likely Swing High, column 1 -> likely Swing Low (depends on upstream)
        swing_high = np.nan
        swing_low = np.nan
        try:
            swing_high = self.swing_hl[0][-1]
            swing_low  = self.swing_hl[1][-1]
        except Exception:
            # fallback safe access
            if isinstance(self.swing_hl, np.ndarray) and self.swing_hl.size:
                arr = _swings_to_transposed_array(self.swing_hl)
                if arr.shape[0] >= 2 and arr.shape[1] >= 1:
                    swing_high = arr[0, -1]
                    swing_low  = arr[1, -1]

        print(swing_high, swing_low)

        # if self.data.Close[-1] > self.data.Open[-1]:
        #     print(f"Index: {self.data.index[-1]}, bul/bear: {self.fvg[0][-1]}, Top: {self.fvg[1][-1]}, Bottom: {self.fvg[2][-1]}, mitig_index: {self.fvg[3][-1]}")

        # print(f"bos: {self.bos[0][-1]}")

        # print(f"index: {self.data.index[-1]}, OB(l/sh): {self.ob[0][-1]}, Top,Bot: {self.ob[1][-1]}, {self.ob[2][-1]}, OBVol: {self.ob[3][-1]}, Perc: {self.ob[4][-1]}")

        # if np.isnan(self.liq[0][-1]):
        #     return
        # print(f"index: {self.data.index[-1]}, liq: {self.liq[0][-1]}, level: {self.liq[1][-1]}, end: {self.liq[2]}, swept: {self.liq[3]}")

        # print(f"index: {self.data.index[-1]}, prevhigh: {self.prev_h[-1]}, prevlow: {self.prev_l[-1]}")

        # print(self.sess[0][-1], self.sess[1][-1], self.sess[-1][-1])

        # print(self.retr[0][-1], self.retr[1][-1], self.retr[-1][-1])

# --- Run backtest ---
if __name__ == "__main__":
    bt = Backtest(df, SMC, cash=10000, commission=0.001)
    stats = bt.run()
    print(stats)

    # Interactive Bokeh plot
    short_ssma = getattr(stats['_strategy'], 'short_ssma_length', "None")
    long_ssma  = getattr(stats['_strategy'], 'long_ssma_length', "None")
    bt.plot(filename=f'plots/{csv_data_name}_shortma_{short_ssma}_longma_{long_ssma}', resample=False, plot_volume=False, open_browser=False)

    # --- MPLFinance plot for detailed inspection (robust handling of swings output) ---
    ohlcv = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    ohlcv.columns = ['open','high','low','close','volume']

    swings = smc.swing_highs_lows(ohlcv, swing_length=10)

    # Convert swings to two pd.Series aligned to ohlcv.index
    def swings_to_series(sw, index):
        # If DataFrame with named cols 'High'/'Low' (or lowercase), use them
        if isinstance(sw, pd.DataFrame):
            if 'High' in sw.columns and 'Low' in sw.columns:
                high_s = sw['High']
                low_s  = sw['Low']
            elif 'high' in sw.columns and 'low' in sw.columns:
                high_s = sw['high']
                low_s  = sw['low']
            else:
                # fallback: take first two columns (preserve shape)
                high_s = sw.iloc[:, 0]
                low_s  = sw.iloc[:, 1] if sw.shape[1] > 1 else pd.Series(np.nan, index=sw.index)
            high_s = pd.Series(high_s.values, index=index)
            low_s  = pd.Series(low_s.values, index=index)
            return high_s, low_s

        # If numpy array
        if isinstance(sw, np.ndarray):
            if sw.ndim == 2:
                if sw.shape[0] == len(index) and sw.shape[1] >= 2:
                    # rows align to bars: col0 -> High, col1 -> Low
                    high = sw[:, 0]
                    low  = sw[:, 1]
                elif sw.shape[1] == len(index) and sw.shape[0] >= 2:
                    # transposed: first axis are columns
                    high = sw[0, :]
                    low  = sw[1, :]
                else:
                    # unknown layout -> try best-effort flattening
                    flattened = sw.reshape(-1)
                    if flattened.size >= 2 * len(index):
                        high = flattened[:len(index)]
                        low  = flattened[len(index):2*len(index)]
                    else:
                        high = np.full(len(index), np.nan)
                        low  = np.full(len(index), np.nan)
            else:
                # 1D array
                if sw.size == len(index):
                    high = sw
                else:
                    high = np.full(len(index), np.nan)
                low = np.full(len(index), np.nan)
            return pd.Series(high, index=index), pd.Series(low, index=index)

        # Unknown type -> return NaNs
        return pd.Series(np.nan, index=index), pd.Series(np.nan, index=index)

    swing_high_s, swing_low_s = swings_to_series(swings, ohlcv.index)

    ohlcv['Swing_High'] = swing_high_s
    ohlcv['Swing_Low']  = swing_low_s

    mpf.plot(
        ohlcv,
        type='candle',
        style='charles',
        figsize=(200,10),
        volume=False,
        addplot=[
            mpf.make_addplot(ohlcv['Swing_High'], color='green', width=1.0),
            mpf.make_addplot(ohlcv['Swing_Low'], color='red', width=1.0)
        ],
        title=f"{csv_data_name} - SMC Swings",
        savefig=f"plots/{csv_data_name}_mplfinance.png",
    )
