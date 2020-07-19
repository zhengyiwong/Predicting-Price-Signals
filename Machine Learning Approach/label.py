import pandas as pd


class TripleBarrier:
    def __init__(self, price, vol_span=50, barrier_horizon=5, factors=None, label=0):
        """
        Labels the Data with the Triple Barrier Method
        :param price: closing price
        :param vol_span: look back to dertermine volatility increment threshold
        :param barrier_horizon: represents vertical length (days) for barrier
        :param factors: repreesnts scalar for barrier height
        :param label: 0 represents label for classification [-1, 0, 1], 1 represenst label for regression -1 <= x <= 1
        """
        self.label = label
        if factors is None:
            factors = [2, 2]
        daily_vol = self.get_daily_vol(prices=price, lookback=vol_span)
        vertical_barriers = self.add_vertical_barrier(
            close=price, num_days=barrier_horizon
        )
        triple_barrier_events = self.get_events(
            close=price,
            factor=factors,
            target=daily_vol,
            vertical_barrier=vertical_barriers,
        )
        self.labels = self.get_labels(triple_barrier_events, price)

    @staticmethod
    def get_daily_vol(prices, lookback=50):
        """
        Daily Volatility Estimates
        Computes the daily volatility at intraday estimation points, applying a span of lookback days to an
        exponentially weighted moving standard deviation.

        This function is used to compute dynamic thresholds for profit taking and stop loss limits
        """
        # find the timestamps at [t-1]
        df = prices.index.searchsorted(prices.index - pd.Timedelta(days=1))
        df = df[df > 0]
        # align timestamps of [t-1] to timestamp p
        df = pd.Series(
            prices.index[df - 1], index=prices.index[prices.shape[0] - df.shape[0] :]
        )
        # get value by tiemstamps
        df = prices.loc[df.index] / prices.loc[df.values].values - 1
        # estimate rolling std
        df = df.ewm(span=lookback).std()
        return df

    @staticmethod
    def add_vertical_barrier(close, num_days=0):
        """
        Adds the vertical barrier
        For each index in events, find the timestamp of the next bar at or immediately after a number of days.
        This function creates a series that has all the timestamps of when the vertical barrier is reached.
        """
        timedelta = pd.Timedelta("{} days".format(num_days))
        nearest_index = close.index.searchsorted(close.index + timedelta)
        nearest_index = nearest_index[nearest_index < close.shape[0]]
        nearest_timestamp = close.index[nearest_index]
        return pd.Series(
            data=nearest_timestamp, index=close.index[: nearest_timestamp.shape[0]]
        )

    @staticmethod
    def touch_barrier(close, events, factor, dates):
        """
        This function applies the triple-barrier. It works on a set of datetime index values.
        Mainly it returns a DataFrame of timestamps regarding the time when the first barriers were reached.
        """
        # Apply Stop Loss / Profit Taking, if it takes place before t1 (end of event)
        events_ = events.loc[dates]
        out = events_[["t1"]].copy(deep=True)
        # Profit taking active
        if factor[0] > 0:
            profit_taking = factor[0] * events_["trgt"]
        else:
            profit_taking = pd.Series(index=events.index)
        # Stop loss active
        if factor[1] > 0:
            stop_loss = -factor[1] * events_["trgt"]
        else:
            stop_loss = pd.Series(index=events.index)
        out["pt"] = pd.Series(dtype=events.index.dtype)
        out["sl"] = pd.Series(dtype=events.index.dtype)
        # Get events
        for loc, vertical_barrier in events_["t1"].fillna(close.index[-1]).iteritems():
            closing_prices = close[loc:vertical_barrier]
            cum_returns = (closing_prices / close[loc] - 1) * events_.side[loc]
            out.at[loc, "sl"] = cum_returns[cum_returns < stop_loss[loc]].index.min()
            out.at[loc, "pt"] = cum_returns[
                cum_returns > profit_taking[loc]
            ].index.min()
        return out

    def get_events(self, close, factor, target, vertical_barrier, side_prediction=None):
        """
        Getting the Time of the First Touch for Triple Barrier Method
        """
        if side_prediction is None:
            side_ = pd.Series(1.0, index=target.index)
            factors = [factor[0], factor[0]]
        else:
            side_ = side_prediction.reindex(close.index)
            factors = factor[:2]
        events = pd.concat(
            {"t1": vertical_barrier, "trgt": target, "side": side_}, axis=1
        )
        events = events.dropna(subset=["trgt"])
        first_touch_dates = self.touch_barrier(
            close=close, events=events, factor=factors, dates=events.index
        )
        for ind in events.index:
            events.at[ind, "t1"] = first_touch_dates.loc[ind, :].dropna().min()
        if side_prediction is None:
            events = events.drop("side", axis=1)
        events["pt"] = factor[0]
        events["sl"] = factor[1]
        return events

    def barrier_touched(self, out_df, events, close):
        """
        Top Horizontal Barrier: 1
        Bottom Horizontal Barrier: -1
        Vertical Barrier: 0
        """
        store = []
        for date_time, values in out_df.iterrows():
            ret = values["ret"]
            target = values["trgt"]
            initial_price = close[close.index == date_time].values[0]
            top_barrier = (
                initial_price + initial_price * events.loc[date_time, "pt"] * target
            )
            btm_barrier = (
                initial_price - initial_price * events.loc[date_time, "sl"] * target
            )
            pt_level_reached = ret > top_barrier
            sl_level_reached = ret < btm_barrier
            if pt_level_reached:
                # Top Barrier Reached
                store.append(1)
            elif sl_level_reached:
                # Bottom Barrier Reached
                store.append(-1)
            else:
                # Vertical Barrier Reached
                if self.label == 0:
                    store.append(0)
                elif self.label == 1:
                    store.append(
                        max(
                            [
                                (ret - initial_price) / (top_barrier - initial_price),
                                (ret - initial_price) / (initial_price - btm_barrier),
                            ],
                            key=abs,
                        )
                    )
        out_df["bin"] = store
        return out_df

    def get_labels(self, triple_barrier_events, close):
        """
        Labels each data point with its respective Triple Barrier Label.
        The ML Algorithm will be trained on the predicted labels.
        """
        events_ = triple_barrier_events.dropna(subset=["t1"])
        all_dates = events_.index.union(other=events_["t1"].array).drop_duplicates()
        prices = close.reindex(all_dates, method="bfill")
        out_df = pd.DataFrame(index=events_.index)
        out_df["ret"] = prices.loc[events_["t1"].array].array
        out_df["trgt"] = events_["trgt"]
        out_df = self.barrier_touched(out_df, triple_barrier_events, close)
        out_df["ret"] = (
            prices.loc[events_["t1"].array].array / prices.loc[events_.index] - 1
        )
        tb_cols = triple_barrier_events.columns
        if "side" in tb_cols:
            out_df["side"] = triple_barrier_events["side"]
        return out_df
