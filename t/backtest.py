import pandas as pd
import plotly.tools as tls
from plotly.offline import iplot


class backtest:
    def __init__(self, base, close, start, period, positions_dict):
        if isinstance(close, pd.Series):
            close = pd.DataFrame(close)
        start = pd.to_datetime(start)
        self.dc = dca(base, close, start, period)
        self.bh = buyandhold(base, close, start, period)
        self.strats = dict()
        for ind, values in positions_dict.items():
            self.strats[ind] = strategy(base, close, values, start, period)

    def get_strat_stats(self, name):
        if name == "DollarCostAvg":
            return self.dc
        elif name == "BuyandHold":
            return self.bh
        else:
            return self.strats[name]

    def get_values(self):
        all_values = pd.concat([self.dc.TotalValue, self.bh.TotalValue], axis=1)
        all_values.columns = ["DollarCostAvg", "BuyandHold"]
        for i in self.strats.keys():
            all_values[i] = self.strats[i].TotalValue
        return all_values

    def get_plot(self):
        d1 = self.get_values()
        d1.sort_index(ascending=True, inplace=True)
        fig = tls.make_subplots(rows=2, cols=1, shared_xaxes=True)
        close = pd.DataFrame(self.dc.Close)
        avgs = d1.columns
        for col in avgs:
            fig.append_trace(
                {"x": d1.index, "y": d1[col], "type": "scatter", "name": col}, 1, 1
            )
        for col in ["Close"]:
            fig.append_trace(
                {
                    "x": close.index,
                    "y": close[col],
                    "type": "scatter",
                    "name": "Closing Price",
                    "line": {"color": "#01DF3A"},
                },
                2,
                1,
            )
        fig["layout"].update(
            height=500,
            title="Relationship between different Strategies",
            paper_bgcolor="#FFF9F5",
            plot_bgcolor="#FFF9F5",
        )
        iplot(fig, filename="pandas/mixed-type subplots")
        return self


def buyandhold(base, close, start, period):
    end = start + pd.offsets.DateOffset(days=period)
    d1 = close.sort_index(ascending=True)[start:end]
    d1.columns = ["Close"]
    d1["Amount_Invested"] = 0
    d1.loc[d1.head(1).index, "Amount_Invested"] = base
    d1["TotalAmountInvested"] = d1["Amount_Invested"].cumsum()
    d1["UnitInvested"] = d1["Amount_Invested"] / d1["Close"]
    d1["TotalUnitInvested"] = d1["UnitInvested"].cumsum()
    d1["TotalValue"] = d1["Close"] * d1["TotalUnitInvested"]
    d1["Gain/Loss"] = d1["TotalValue"] - d1["TotalAmountInvested"]
    return d1


def dca(base, close, start, period):
    end = start + pd.offsets.DateOffset(days=period)
    d1 = close.sort_index(ascending=True)[start:end]
    d1.columns = ["Close"]
    d1["Amount_Invested"] = base / len(d1)
    d1["TotalAmountInvested"] = d1["Amount_Invested"].cumsum()
    d1["UnitInvested"] = d1["Amount_Invested"] / d1["Close"]
    d1["TotalUnitInvested"] = d1["UnitInvested"].cumsum()
    d1["TotalValue"] = d1["Close"] * d1["TotalUnitInvested"]
    d1["Gain/Loss"] = d1["TotalValue"] - d1["TotalAmountInvested"]
    return d1


def strategy(base, close, buy_signal, start, period):
    end = start + pd.offsets.DateOffset(days=period)
    d1 = close.sort_index(ascending=True)[start:end]
    d1.columns = ["Close"]
    d1["Signal"] = buy_signal[start:end]
    Signal = []
    TotalValues = []
    CashHelds = []
    UnitsInvested = []
    UnitsHeld = []
    for row in d1.itertuples(index=True, name="Pandas"):
        if len(Signal) == 0:
            Amount_Invested = base
            UnitInvested = Amount_Invested / row.Close
            UnitsHeld = UnitInvested
            CashHeld = 0
            TotalValue = UnitInvested * row.Close
        elif (row.Signal == 1 and Signal[-1] == 1) or (
            row.Signal == 1 and len(Signal) == 1
        ):
            CashHeld = 0
            UnitInvested = UnitsHeld
            TotalValue = UnitInvested * row.Close
        elif row.Signal == 1 and Signal[-1] != 1:
            Amount_Invested = CashHelds[-1]
            UnitInvested = Amount_Invested / row.Close
            UnitsHeld = UnitInvested
            CashHeld = 0
            TotalValue = UnitInvested * row.Close
        elif row.Signal != 1 and Signal[-1] != 1 and len(Signal) > 1:
            UnitInvested = 0
            CashHeld = CashHelds[-1]
            TotalValue = 0
        elif (row.Signal != 1 and Signal[-1] == 1) or (
            row.Signal != 1 and len(Signal) == 1
        ):
            UnitInvested = 0
            CashHeld = UnitsHeld * row.Close
            UnitsHeld = 0
            TotalValue = 0
        Signal.append(row.Signal)
        TotalValues.append(TotalValue)
        UnitsInvested.append(UnitInvested)
        CashHelds.append(CashHeld)
    d1["TotalUnitInvested"] = UnitsInvested
    d1["CashHeld"] = CashHelds
    d1["TotalInvestment"] = TotalValues
    d1["TotalValue"] = d1["CashHeld"] + d1["TotalInvestment"]
    d1["Gain/Loss"] = d1["TotalValue"] - base
    return d1
