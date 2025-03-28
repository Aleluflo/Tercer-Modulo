import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import ta
from datos import data,rsi,bb,ema,dataset,rf,N

def sharpe_ratio(dataset, rf, N):
    returns = dataset["Close"].pct_change().dropna()
    mean = returns.mean()
    std = returns.std()

    sharpe_ratio = np.sqrt(N) * (mean / std)

    return sharpe_ratio


sharpe = sharpe_ratio(dataset, rf, N)
print(f"Ratio de Sharpe: {sharpe:.4f}")


# Sortino

def sortino_ratio(dataset, N,rf):
    returns = dataset["Close"].pct_change().dropna()
    mean_excess_return = (returns.mean() - rf) * N
    downside_std = returns[returns < 0].std() * np.sqrt(N)

    # Evitar división por cero
    if downside_std == 0:
        return np.nan

    sortino_ratio = mean_excess_return / downside_std

    return sortino_ratio


sortino = sortino_ratio(dataset, N, rf)
print(f"Ratio de Sortino: {sortino:.4f}")


# Calmar

def calmar_ratio(dataset, N, rf):
    returns = dataset["Close"].pct_change().dropna()
    annualized_return = (returns.mean() - rf) * N

    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    dd = (cumulative_returns / peak) - 1
    max_dd = dd.min()

    if max_dd == 0:
        return np.nan

    calmar_ratio = annualized_return / abs(max_dd)

    return calmar_ratio


calmar = calmar_ratio(dataset, N, rf)
print(f"Ratio de Calmar: {calmar:.4f}")


# Win/Loss

def win_loss_percentage(dataset):
    returns = dataset["Close"].pct_change().dropna()
    wins = (returns > 0).sum()
    total_trades = len(returns)

    if total_trades == 0:
        return np.nan

    win_loss_ratio = (wins / total_trades) * 100

    return win_loss_ratio


win_loss = win_loss_percentage(dataset)
print(f"Win/Loss Percentage: {win_loss:.2f}%")