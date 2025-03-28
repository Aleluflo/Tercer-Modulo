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



