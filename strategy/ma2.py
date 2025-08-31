from backtesting import Strategy, Backtest
import pandas as pd
import pandas_ta as ta
import numpy as np
from backtesting.lib import plot_heatmaps, crossover, barssince, FractionalBacktest, resample_apply
from smartmoneyconcepts import smc

smc.bos_choch()
csv_data_name = "btcusd_5"
df = pd.read_csv(f'../csv_data/csv_{csv_data_name}.csv', parse_dates=True, index_col=[0])

# print(df.head())
df.info()


# --- Helper to build OHLCV DataFrame (for smc library) ---
def _prepare_ohlcv(data):
    return pd.DataFrame({
        'open':   data.Open.s,
        'high':   data.High.s,
        'low':    data.Low.s,
        'close':  data.Close.s,
        'volume': data.Volume.s
    }, index=data.index)

def session_open(data, session, start_time=None, end_time=None, time_zone="UTC"):
    """1 if in session, else 0"""
    ohlc = _prepare_ohlcv(data)
    df = smc.sessions(ohlc, session=session, start_time=start_time, end_time=end_time, time_zone=time_zone)
    return df.to_numpy().T


# other
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

def macd(close_data, fast, slow, signal):
    m = ta.macd(close=close_data, fast=fast, slow=slow, signal=signal)
    macd_col = f"MACD_{fast}_{slow}_{signal}"
    signal_col = f"MACDs_{fast}_{slow}_{signal}"
    hist_col = f"MACDh_{fast}_{slow}_{signal}"
    return m[macd_col].to_numpy().T, m[signal_col].to_numpy().T, m[hist_col].to_numpy().T




# other
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

def macd(close_data, fast, slow, signal):
    m = ta.macd(close=close_data, fast=fast, slow=slow, signal=signal)
    macd_col = f"MACD_{fast}_{slow}_{signal}"
    signal_col = f"MACDs_{fast}_{slow}_{signal}"
    hist_col = f"MACDh_{fast}_{slow}_{signal}"
    return m[macd_col].to_numpy().T, m[signal_col].to_numpy().T, m[hist_col].to_numpy().T


class Ma_macd(Strategy):
    atr_length = 14
    atr_coe = 2 # for sl, tp calculation
    rrr = 3.2
    # session
    session_name = "New York"
    # start_time = "06:30"
    # end_time = "16:30"
    session_tz = "UTC"
    short_ssma_length = 50
    short_ema_length = 100
    long_ssma_length = 200
    long_ema_length = 200
    macd_fast_length = 12
    macd_slow_length = 26
    macd_signal_length = 9


    def init(self):
        self.session_ny = self.I(session_open, self.data, session=self.session_name, time_zone='UTC')
        self.atr = self.I(atr, pd.DataFrame({
                'High': self.data.High.s,
                'Low': self.data.Low.s,
                'Close': self.data.Close.s
            }), self.atr_length)

        self.macd_line, self.macd_signal, self.macd_hist = self.I(
            macd, self.data.Close.s, self.macd_fast_length, self.macd_slow_length, self.macd_signal_length
        )

        self.short_ssma = self.I(ssma, self.data.Close.s, self.short_ssma_length)
        self.long_ssma = self.I(ssma, self.data.Close.s, self.long_ssma_length)



    def next(self):
        # sl, tp calculation
        sl_long = self.data.Close[-1] - self.atr_coe * self.atr
        sl_short = self.data.Close[-1] + self.atr_coe * self.atr
        tp_long = self.data.Close[-1] + self.rrr * self.atr_coe * self.atr
        tp_short = self.data.Close[-1] - self.rrr * self.atr_coe * self.atr

        # MACD crossover check
        macd_bullish_crossover = False
        macd_bearish_crossover = False
        if crossover(self.macd_signal, self.macd_line):
            macd_bullish_crossover = True
        elif crossover(self.macd_line, self.macd_signal):
            macd_bearish_crossover = True


        if (
            self.short_ssma[-1] > self.long_ssma[-1] and
            macd_bullish_crossover and not self.position
        ):
            self.buy(size=.1, sl=sl_long, tp=tp_long)
        elif (
            self.long_ssma[-1] > self.short_ssma[-1] and
            macd_bearish_crossover and not self.position
        ):
            self.sell(size=.1, sl=sl_short, tp=tp_short)

        # if self.position.is_long:
        #     if crossover(self.long_ssma, self.short_ssma):
        #         self.position.close()
        # elif self.position.is_short:
        #     if crossover(self.short_ssma, self.long_ssma):
        #         self.position.close()


if __name__ == "__main__":
    bt = FractionalBacktest(df, Ma_macd, cash=100, commission=.003, finalize_trades=True)
    # stats, heatmap = bt.optimize(
    #     short_ema_5m=range(5, 26),
    #     long_ema_5m=range(14, 36),
    #     constraint = lambda param: param.short_ema_5m < param.long_ema_5m,
    #     return_heatmap=True,
    # )

    # plot_heatmaps(heatmap)
    # print(stats)


    stats = bt.run()
    short_ssma = stats['_strategy'].short_ssma_length
    long_ssma = stats['_strategy'].long_ssma_length
    print(stats)
    bt.plot(filename=f'plots/{csv_data_name}_shortma_{short_ssma}_longma_{long_ssma}', resample=False, plot_volume=False)


