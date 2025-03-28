import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import ta
from datos import data,rsi,bb,ema,dataset,rf,N,capital,com,n_shares,portfolio_value,stop_loss,take_profit,win,losses,active_long_positions,active_short_positions
from objective_func import objective_func


for i, row in dataset.iterrows():
    if active_long_positions:

        if row.Close < active_long_positions["stop_loss"]:
            pnl = row.Close * n_shares * (1 - com)
            capital += pnl
            active_long_positions = None
            losses += 1


        elif row.Close > active_long_positions["take_profit"]:
            pnl = row.Close * n_shares * (1 - com)
            capital += pnl
            active_long_positions = None
            win += 1

    if active_short_positions:

        if row.Close > active_short_positions["stop_loss"]:
            pnl = (active_short_positions["opened_at"] - row.Close) * n_shares * (1 - com)
            capital += pnl
            active_short_positions = None
            losses += 1


        elif row.Close < active_short_positions["take_profit"]:
            pnl = (active_short_positions["opened_at"] - row.Close) * n_shares * (1 - com)
            capital += pnl
            active_short_positions = None
            win += 1

    if (row.RSI_BUY + row.BB_BUY + row.EMA_BUY) >= 2 and active_long_positions is None:
        cost = row.Close * n_shares * (1 + com)
        if capital >= cost:
            capital -= cost
            active_long_positions = {
                "datetime": row.Datetime,
                "opened_at": row.Close,
                "take_profit": row.Close * (1 + take_profit),
                "stop_loss": row.Close * (1 - stop_loss),
            }

    if (row.RSI_SELL + row.BB_SELL + row.EMA_SELL) >= 2 and active_short_positions is None:
        credit = row.Close * n_shares * (com)
        if capital >= credit:
            capital -= credit
            active_short_positions = {
                "datetime": row.Datetime,
                "opened_at": row.Close,
                "take_profit": row.Close * (1 - take_profit),
                "stop_loss": row.Close * (1 + stop_loss),
            }

    long_value = row.Close * n_shares if active_long_positions else 0
    short_value = (active_short_positions["opened_at"] - row.Close) * n_shares if active_short_positions else 0

    portfolio_value.append(capital + long_value + short_value)

# Plot

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.plot(portfolio_value, label="Porfolio Value")
ax.legend()
ax2 = ax.twinx()
ax2.plot(data.Close, c="C1")
plt.title("Portfolio Value vs Stock Price")

plt.show()

# Optimization

study = optuna.create_study(direction="maximize")
study.optimize(lambda x: objective_func(x,data), n_trials=50)

# Best Values
study.best_value
study.best_params

# Optimized Sharpe

rsi = ta.momentum.RSIIndicator(data.Close, window=57)
bb = ta.volatility.BollingerBands(data.Close, window=63, window_dev=1)
ema = ta.trend.EMAIndicator(data.Close, window=48)
dataset = data.copy()

dataset["RSI"] = rsi.rsi()
dataset["BB"] = bb.bollinger_mavg()
dataset["EMA"] = ema.ema_indicator()

dataset["RSI_BUY"] = dataset["RSI"] < 19
dataset["RSI_SELL"] = dataset["RSI"] > 89

dataset["BB_BUY"] = bb.bollinger_lband_indicator().astype(bool) # lower band
dataset["BB_SELL"] = bb.bollinger_hband_indicator().astype(bool) # higher band

dataset["EMA_BUY"] = dataset["Close"] > dataset["EMA"]
dataset["EMA_SELL"] = dataset["Close"] < dataset["EMA"]

dataset = dataset.dropna()

capital = 1000000
com = 0.125 / 100  # Comisión de compra/venta
n_shares = 4000  # Número de acciones por operación

portfolio_value = [capital]

stop_loss = 0.0406  # Stop Loss del 15%
take_profit = 0.0815  # Take Profit del 8%

win = 0
losses = 0

active_long_positions = None  # Solo una posición activa de cada tipo
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

fig, ax = plt.subplots(1, 1, figsize=(12,6))

ax.plot(portfolio_value,label = "Porfolio Value")
ax.legend()
ax2 = ax.twinx()
ax2.plot(data.Close, c="C1")
plt.title("Portfolio Value vs Stock Price with an Optimized Sharpe")

plt.show()



