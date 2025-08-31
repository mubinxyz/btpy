# strategy/smc_test.py

from backtesting import Strategy, Backtest
import pandas as pd
import pandas_ta as ta
import numpy as np
from backtesting.lib import plot_heatmaps, crossover, barssince, FractionalBacktest, resample_apply
from smartmoneyconcepts import smc

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

# --- Named wrappers for SMC indicators ---

def fvg_signal(data, join_consecutive=False):
    """1 if bullish FVG, -1 if bearish, 0 none"""
    ohlc = _prepare_ohlcv(data)
    df = smc.fvg(ohlc, join_consecutive=join_consecutive)
    return df.to_numpy().T


def swing_highs_lows_series(data, swing_length=50):
    """Returns swings DataFrame for use inside next()"""
    ohlc = _prepare_ohlcv(data)
    return smc.swing_highs_lows(ohlc, swing_length=swing_length).to_numpy().T


def bos_signal(data, swing_length=50, close_break=True):
    """1 bullish BOS, -1 bearish, 0 none"""
    ohlc = _prepare_ohlcv(data)
    swings = smc.swing_highs_lows(ohlc, swing_length=swing_length) # this is due to self.I that doesn't call 2 functions.
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
    swing_length = 4
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
        # print(self.fvg[0][-1])

        print(self.swing_hl[0][-1], self.swing_hl[1][-1])

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
    short_ssma = stats['_strategy'].short_ssma_length
    long_ssma = stats['_strategy'].long_ssma_length
    bt.plot(filename=f'plots/{csv_data_name}_shortma_{short_ssma}_longma_{long_ssma}', resample=False, plot_volume=False)