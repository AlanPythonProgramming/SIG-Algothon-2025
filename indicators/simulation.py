import pandas as pd
import numpy as np
from hurst import compute_Hc
import matplotlib.pyplot as plt

df = pd.read_csv("prices.txt", delim_whitespace=True, header=None)
stock_list = [11, 12, 22, 23, 26, 29, 46]

short_ema = 20
long_ema = 100
n_simulations = 10000
n_days = 100
low_return = -0.01
high_return = 0.01

np.random.seed(42)

print("Simulated EMA strategy performance over 1000 paths:")
print("-" * 60)

for i in stock_list:
    base_price = df[i].dropna().iloc[-1]
    strategy_returns = []
    buyhold_returns = []

    for _ in range(n_simulations):
        # Simulate returns and build price path
        sim_returns = np.random.uniform(low=low_return, high=high_return, size=n_days)
        sim_prices = [base_price]
        for r in sim_returns:
            sim_prices.append(sim_prices[-1] * (1 + r))
        price_series = pd.Series(sim_prices)

        # Build DataFrame with strategy logic
        data = pd.DataFrame({'price': price_series})
        data['EMA_short'] = data['price'].ewm(span=short_ema, adjust=False).mean()
        data['EMA_long'] = data['price'].ewm(span=long_ema, adjust=False).mean()

        data['position'] = 0
        data.loc[data['EMA_short'] > data['EMA_long'], 'position'] = 1
        data.loc[data['EMA_short'] < data['EMA_long'], 'position'] = -1
        data['position'] = data['position'].shift(1)

        data['returns'] = data['price'].pct_change()
        data['strategy'] = data['position'] * data['returns']

        data['cumulative_strategy'] = (1 + data['strategy']).cumprod()
        data['cumulative_price'] = (1 + data['returns']).cumprod()

        strategy_returns.append(data['cumulative_strategy'].iloc[-1])
        buyhold_returns.append(data['cumulative_price'].iloc[-1])

    avg_strategy_return = np.mean(strategy_returns)
    avg_buyhold_return = np.mean(buyhold_returns)

    print(f"Asset {i}: Avg EMA Strategy = {avg_strategy_return:.4f}, Avg Buy & Hold = {avg_buyhold_return:.4f}")
