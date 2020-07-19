import pandas as pd
import numpy as np


# Momentum Indicators
# Momentum indicators are tools utilized by traders to get a better understanding of the speed or rate at which the
# price of a security changes. Momentum indicators are best used with other indicators and tools because they don’t work
# to identify the direction of movement, only the timeframe in which the price change is occurring.


def rsi(close: pd.Series, rsi_period=14):
    """
    Measures recent trading strength, velocity of change in the trend, and magnitude of the move.
    """
    delta = close.diff(1)
    data = pd.DataFrame(delta)
    gain = delta.mask(delta < 0, 0)
    data["gain"] = gain
    loss = delta.mask(delta > 0, 0)
    data["loss"] = loss
    avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    data["rsi"] = 100 - (100 / (1 + abs(avg_gain / avg_loss)))
    return data["rsi"]


def stochastic_osc(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period, d_period
):
    """
    Used to predict price turning points by comparing the closing price to its price range.
    k : Period difference for PercentK
    d : Length of moving average in periods for PercentD
    """
    data = pd.DataFrame()
    data["stok"] = np.nan
    for i in range(len(high)):
        index = high.index[i]
        if i < k_period:
            highrange = high[0:k_period]
            lowrange = low[0:k_period]
            data.loc[index, "stok"] = (
                (close[i] - lowrange.min()) / (highrange.max() - lowrange.min())
            ) * 100
        else:
            highrange = high[i + 1 - k_period : i + 1]
            lowrange = low[i + 1 - k_period : i + 1]
            data.loc[index, "stok"] = (
                (close[i] - lowrange.min()) / (highrange.max() - lowrange.min())
            ) * 100
    STOD = data["stok"].ewm(span=d_period, min_periods=d_period, adjust=False).mean()
    return STOD


def williamsR(high: pd.Series, low: pd.Series, close: pd.Series):
    """
    The Williams Percent Range, also called Williams %R, is a momentum indicator that shows you where the last closing
    price is relative to the highest and lowest prices of a given time period. As an oscillator, Williams %R tells
    you when a currency pair might be “overbought” or “oversold.”
    """
    return (high.max() - close) / (high - low.min()) * (-100)


def evm(high: pd.Series, low: pd.Series, volume: pd.Series):
    """
    Richard Arms' Ease of Movement indicator is a technical study that attempts to quantify a mix of momentum and
    volume information into one value.
    """
    dm = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)
    br = (volume / 100000000) / (high - low)
    return dm / br


def awesome_osc(high: pd.Series, low: pd.Series, high_period=34, low_period=5):
    """
    The Awesome Oscillator (AO) is an indicator used to measure market momentum. AO calculates the difference between a
    34 Period and 5 Period Simple Moving Average.

    """
    fast_period = (
        high.rolling(window=high_period).mean() + low.rolling(window=high_period).mean()
    ) / 2
    slow_period = (
        high.rolling(window=low_period).mean() + low.rolling(window=low_period).mean()
    ) / 2
    return fast_period - slow_period


def coppock_ind(values: pd.Series):
    """
    The Coppock Curve (CC) was introduced by economist Edwin Coppock.﻿ While useful, the indicator is not commonly
    discussed among traders and investors. Traditionally used to spot long-term trend changes in major stock indexes,
    traders can use the indicator for any time and in any market to isolate potential trend shifts and generate trade
    signals.
    """
    ROC1 = values / values.shift(14) - 1
    ROC2 = values / values.shift(11) - 1
    ROC3 = ROC1 + ROC2
    ROC = ROC3
    list_of_values = []
    tmp2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ROC = ROC.fillna(int(999))
    ROC.rolling(window=10).apply(
        lambda x: list_of_values.append(x.values) or 999, raw=False
    )
    buyCoppock = [np.nan] * 9
    for i in list_of_values:
        if 999 in i:
            buyCoppock.append(np.nan)
        else:
            buyCoppock.append(sum(np.multiply(i, tmp2)))
    return buyCoppock


def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series):
    """
    The Ichimoku Cloud, also known as Ichimoku Kinko Hyo, is a versatile indicator that defines support and resistance,
    identifies trend direction, gauges momentum and provides trading signals.
    """
    # Tenkan-sen (Conversion Line)
    nine_period_high = high.rolling(window=9).max()
    nine_period_low = low.rolling(window=9).min()
    tenkan_sen = (nine_period_high + nine_period_low) / 2
    # Kijun-sen (Base Line)
    twenty_six_period_high = high.rolling(window=26).max()
    twenty_six_period_low = low.rolling(window=26).min()
    kijun_sen = (twenty_six_period_high + twenty_six_period_low) / 2
    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    # Senkou Span B (Leading Span B)
    fifty_two_period_high = high.rolling(window=52).max()
    fifty_two_period_low = low.rolling(window=52).min()
    senkou_span_b = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)
    # Chikou Span (Closing Price)
    chikou_span = close.shift(26)
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span


# Volatility Indicators
# These technical indicators measure the rate of price movement, regardless of direction.


def bollinger_bands(close: pd.Series, window, std_scalar):
    """
    Bollinger bands: Measures the “highness” or “lowness” of price, relative to previous trades.
    """
    rolling_mean = close.ewm(span=window).mean()
    rolling_std = close.ewm(span=window).mean()
    upper_band = rolling_mean + (rolling_std * std_scalar)
    lower_band = rolling_mean - (rolling_std * std_scalar)
    return rolling_mean, upper_band, lower_band


# Trend Indicator
# These technical indicators measure the direction and strength of a trend by comparing prices to an established
# baseline.


def daily_ma(values: pd.Series, days=3):
    return values.shift(1).rolling(window=days).mean()


def macd(values: pd.Series):
    """
    Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security's price. The MACD is calculated by subtracting the 26-period Exponential
    Moving Average (EMA) from the 12-period EMA.
    """
    macd_value = (
        values.ewm(span=12, min_periods=12, adjust=False).mean()
        - values.ewm(span=26, min_periods=26, adjust=False).mean()
    )
    macdsignalline = macd_value.ewm(span=9, min_periods=9, adjust=False).mean()
    return macd_value, macdsignalline
