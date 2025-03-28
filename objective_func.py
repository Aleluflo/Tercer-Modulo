import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import ta
from utils import sharpe_ratio,sortino_ratio,calmar_ratio,win_loss_percentage
from datos import N,rf


def objective_func(trial, data):
    rsi_window = trial.suggest_int("rsi_window", 10, 100)
    rsi_lower = trial.suggest_int("rsi_lower", 10, 25)
    rsi_upper = trial.suggest_int("rsi_upper", 70, 95)

    stop_loss = trial.suggest_float("stop_loss", 0.01, 0.1)
    take_profit = trial.suggest_float("take_profit", 0.05, 0.2)
    n_shares = trial.suggest_categorical("n_shares", [1000, 2000, 4000, 5000])

    bb_window = trial.suggest_int("bb_window", 5, 100)
    bb_window_dev = trial.suggest_int("bb_window_dev", 1, 3)
    ema_window = trial.suggest_int("ema_window", 10, 55)

    rsi = ta.momentum.RSIIndicator(data.Close, window=rsi_window)
    bb = ta.volatility.BollingerBands(data.Close, window=bb_window,
                                      window_dev=bb_window_dev)  # dos desviaciones estandar
    ema = ta.trend.EMAIndicator(data.Close, window=ema_window)
    dataset = data.copy()
    dataset["RSI"] = rsi.rsi()
    dataset["BB"] = bb.bollinger_mavg()
    dataset["EMA"] = ema.ema_indicator()

    dataset["RSI_BUY"] = dataset["RSI"] < rsi_lower
    dataset["RSI_SELL"] = dataset["RSI"] > rsi_upper

    dataset["BB_BUY"] = bb.bollinger_lband_indicator().astype(bool)  # lower band
    dataset["BB_SELL"] = bb.bollinger_hband_indicator().astype(bool)  # higher band

    dataset["EMA_BUY"] = dataset["Close"] > dataset["EMA"]
    dataset["EMA_SELL"] = dataset["Close"] < dataset["EMA"]

    dataset = dataset.dropna()
    capital = 1000000
    com = 0.125 / 100

    portfolio_value = [capital]

    win = 0
    losses = 0

    active_long_positions = None  # una sola posicion activa de cada tipo
    active_short_positions = None

    for i, row in dataset.iterrows():
        # === Cerrar posiciones largas (Long) ===
        if active_long_positions:
            # Cierre por Stop Loss
            if row.Close < active_long_positions["stop_loss"]:
                pnl = row.Close * n_shares * (1 - com)
                capital += pnl
                active_long_positions = None
                losses += 1

            # Cierre por Take Profit
            elif row.Close > active_long_positions["take_profit"]:
                pnl = row.Close * n_shares * (1 - com)
                capital += pnl
                active_long_positions = None
                win += 1

        # === Cerrar posiciones cortas (Short) ===
        if active_short_positions:
            # Cierre por Stop Loss
            if row.Close > active_short_positions["stop_loss"]:
                pnl = (active_short_positions["opened_at"] - row.Close) * n_shares * (1 - com)
                capital += pnl
                active_short_positions = None
                losses += 1

            # Cierre por Take Profit
            elif row.Close < active_short_positions["take_profit"]:
                pnl = (active_short_positions["opened_at"] - row.Close) * n_shares * (1 - com)
                capital += pnl
                active_short_positions = None
                win += 1

        # === Abrir posiciones largas (Long) ===
        if (row.RSI_BUY + row.BB_BUY + row.EMA_BUY) >= 2 and active_long_positions is None:
            cost = row.Close * n_shares * (1 + com)
            if capital >= cost:  # Solo abrir si hay capital suficiente
                capital -= cost
                active_long_positions = {
                    "datetime": row.Datetime,
                    "opened_at": row.Close,
                    "take_profit": row.Close * (1 + take_profit),
                    "stop_loss": row.Close * (1 - stop_loss),
                }

        # === Abrir posiciones cortas (Short) ===
        if (row.RSI_SELL + row.BB_SELL + row.EMA_SELL) >= 2 and active_short_positions is None:
            credit = row.Close * n_shares * (com)
            if capital >= credit:  # Solo abrir si hay capital suficiente
                capital -= credit
                active_short_positions = {
                    "datetime": row.Datetime,
                    "opened_at": row.Close,
                    "take_profit": row.Close * (1 - take_profit),
                    "stop_loss": row.Close * (1 + stop_loss),
                }

        # === Calcular el valor del portafolio ===
        long_value = row.Close * n_shares if active_long_positions else 0
        short_value = (active_short_positions["opened_at"] - row.Close) * n_shares if active_short_positions else 0

        # Agregar el valor total del portafolio
        portfolio_value.append(capital + long_value + short_value)
        returns = np.diff(portfolio_value) / portfolio_value[:-1]

    sharpe = sharpe_ratio(portfolio_value, rf, N)
    sortino = sortino_ratio(portfolio_value, N, rf)
    calmar = calmar_ratio(portfolio_value, N, rf)
    win_loss = win_loss_percentage(portfolio_value)

    return sharpe
