import pandas as pd
from hurst import compute_Hc
import matplotlib.pyplot as plt

# Load price data
df = pd.read_csv("prices.txt", delim_whitespace=True, header=None)

short_ema = 10  # Fixed short EMA

print("Optimal long_ema for assets with Hurst > 0.55")
print("-" * 80)

for i in range(50):
    price = df[i].dropna()
    H, _, _ = compute_Hc(price)

    if H > 0:
        best_return = -float('inf')
        best_long_ema = None

        for long_ema in range(50, 201):
            data = pd.DataFrame({'price': price})
            data['EMA_short'] = data['price'].ewm(span=short_ema, adjust=False).mean()
            data['EMA_long'] = data['price'].ewm(span=long_ema, adjust=False).mean()

            # Crossover strategy: long if short > long, short if short < long
            data['position'] = 0
            data.loc[data['EMA_short'] > data['EMA_long'], 'position'] = 1
            data.loc[data['EMA_short'] < data['EMA_long'], 'position'] = -1
            data['position'] = data['position'].shift(1)

            # Calculate strategy returns
            data['returns'] = data['price'].pct_change()
            data['strategy'] = data['position'] * data['returns']
            cumulative_return = (1 + data['strategy']).cumprod().iloc[-1]

            # Update best if this one is better
            if pd.notna(cumulative_return) and cumulative_return > best_return:
                best_return = cumulative_return
                best_long_ema = long_ema

        # Print results for this asset
        if best_long_ema is not None:
            print(f"Asset {i}: H = {H:.3f}, Best long_ema = {best_long_ema}, Max Return = {best_return:.2f}")