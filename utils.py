import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import ta
from datos import data,rsi,bb,ema,dataset,rf,N


def sharpe_ratio(portfolio_val, rf, N):
    ds = pd.Series(portfolio_val)
    returns = ds.pct_change().dropna()
    mean = returns.mean()  # *N
    std = returns.std()  # * np.sqrt(N)
    if std == 0:
        return 0

    sharpe_ratio = np.sqrt(N) * (mean / std)

    return sharpe_ratio



# Sortino

def sortino_ratio(portfolio_val, N, rf):
    returns = pd.Series(portfolio_val).pct_change().dropna()
    mean_excess_return = (returns.mean() - rf) * N
    downside_std = returns[returns < 0].std() * np.sqrt(N)

    # Evitar división por cero
    if downside_std == 0:
        return np.nan

    sortino = mean_excess_return / downside_std

    return sortino


# Calmar

def calmar_ratio(portfolio_val, N, rf):
    returns = pd.Series(portfolio_val).pct_change().dropna()
    annualized_return = (returns.mean() - rf) * N

    cumulative_returns = (1 + returns).cumprod()  # Crecimiento del portafolio
    peak = cumulative_returns.cummax()  # Picos históricos
    dd = (cumulative_returns / peak) - 1  # Drawdown en cada punto
    max_dd = dd.min()  # Máximo drawdown registrado

    if max_dd == 0:
        return np.nan

    calmar_ratio = annualized_return / abs(max_dd)

    return calmar_ratio


# Win/Loss

def win_loss_percentage(portfolio_val):
    returns = pd.Series(portfolio_val).pct_change().dropna()
    wins = (returns > 0).sum()  # Retornos positivos
    total_trades = len(returns)  # Total de operaciones

    if total_trades == 0:
        return np.nan

    win_loss_ratio = (wins / total_trades) * 100

    return win_loss_ratio
