import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import ta

data = pd.read_csv("aapl_5m_train.csv").dropna()
rsi = ta.momentum.RSIIndicator(data.Close, window=25)
bb = ta.volatility.BollingerBands(data.Close, window=16, window_dev=2)
ema = ta.trend.EMAIndicator(data.Close, window=30)

dataset = data.copy()
dataset["RSI"] = rsi.rsi()
dataset["BB"] = bb.bollinger_mavg()
dataset["EMA"] = ema.ema_indicator()

dataset["RSI_BUY"] = dataset["RSI"] < 25
dataset["RSI_SELL"] = dataset["RSI"] > 75

dataset["BB_BUY"] = bb.bollinger_lband_indicator().astype(bool) # lower band
dataset["BB_SELL"] = bb.bollinger_hband_indicator().astype(bool) # higher band

dataset["EMA_BUY"] = dataset["Close"] > dataset["EMA"]
dataset["EMA_SELL"] = dataset["Close"] < dataset["EMA"]

dataset = dataset.dropna()

capital = 1000000
com = 0.125 / 100  # Comisión de compra/venta
n_shares = 1000  # Número de acciones por operación

portfolio_value = [capital]

stop_loss = 0.15  # Stop Loss del 15%
take_profit = 0.08  # Take Profit del 8%

win = 0
losses = 0

active_long_positions = None  # Solo una posición activa de cada tipo
active_short_positions = None

rf = 0
N = 252*78
