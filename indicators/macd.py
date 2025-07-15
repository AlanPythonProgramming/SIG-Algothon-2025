import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import ta

# Load data
df = pd.read_csv("prices.txt", delim_whitespace=True, header=None)

# Compute returns
returns = df.pct_change().dropna()
n = 4

# Calculate MACD
macd = ta.trend.MACD(close=df[n], window_slow=26, window_fast=12, window_sign=9)
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()
df['macd_diff'] = macd.macd_diff()



for i in range(50):
    plt.plot(df[i])
    plt.show()
# # Set up subplots
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# # Plot returns
# ax1.plot(returns.index, returns[n], label='Returns', color='blue')
# ax1.set_title('Asset Returns')
# ax1.set_ylabel('Daily Return')
# ax1.grid(True)
# ax1.legend()

# # Plot MACD components
# ax2.plot(df.index, df['macd'], label='MACD Line', color='black')
# ax2.plot(df.index, df['macd_signal'], label='Signal Line', color='orange')
# ax2.bar(df.index, df['macd_diff'], label='MACD Histogram', color='red', alpha=0.5)
# ax2.set_title('MACD Indicator')
# ax2.set_ylabel('MACD Value')
# ax2.set_xlabel('Time')
# ax2.grid(True)
# ax2.legend()

# plt.tight_layout()
# plt.show()