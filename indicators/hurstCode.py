# import pandas as pd
# from hurst import compute_Hc
# import matplotlib.pyplot as plt

# # Load raw price data
# df = pd.read_csv("prices.txt", delim_whitespace=True, header=None)

# # Set moving average windows
# short_window = 10
# long_window =75
# print("Assets with Hurst > 0.55 and MA strategy performance:")
# print("-" * 60)

# for i in range(50):
#     price = df[i].dropna()

#     # Calculate Hurst exponent
#     H, _, _ = compute_Hc(price)

#     if H > 0.5:
#         # Prepare dataframe
#         data = pd.DataFrame({'price': price})
#         data['MA_short'] = data['price'].rolling(window=short_window).mean()
#         data['MA_long'] = data['price'].rolling(window=long_window).mean()

#         # Generate MA crossover signal
#         data['position'] = 0
#         data.loc[data['MA_short'] > data['MA_long'], 'position'] = 1
#         data.loc[data['MA_short'] <= data['MA_long'], 'position'] = 0
#         data['position'] = data['position'].shift(1)

#         # Compute returns
#         data['returns'] = data['price'].pct_change()
#         data['strategy'] = data['position'] * data['returns']

#         # Cumulative returns
#         data['cumulative_strategy'] = (1 + data['strategy']).cumprod()
#         data['cumulative_price'] = (1 + data['returns']).cumprod()

#         final_strategy_return = data['cumulative_strategy'].iloc[-1]
#         final_price_return = data['cumulative_price'].iloc[-1]

#         print(f"Asset {i}: H = {H:.3f}, Strategy Return = {final_strategy_return:.2f}, Buy & Hold = {final_price_return:.2f}")


import pandas as pd
from hurst import compute_Hc
import matplotlib.pyplot as plt

# Load price data
df = pd.read_csv("prices.txt", delim_whitespace=True, header=None)
df = df[649:749]
# Set EMA windows
short_ema = 20
long_ema = 100

thing = [2,3,4,7,16]
for i in thing:
    plt.plot(df[i])
    plt.show()

print("Assets with Hurst > 0.55 and EMA strategy performance:")
print("-" * 60)



for i in range(50):
    price = df[i].dropna()

    # Compute Hurst exponent
    H, _, _ = compute_Hc(price)
    
    if H > 0:
        print(H)
        # Construct DataFrame with EMAs
        data = pd.DataFrame({'price': price})
        data['EMA_short'] = data['price'].ewm(span=short_ema, adjust=False).mean()
        data['EMA_long'] = data['price'].ewm(span=long_ema, adjust=False).mean()

        # Generate EMA crossover signal
        data['position'] = 0
        data.loc[data['EMA_short'] > data['EMA_long'], 'position'] = 1
        data.loc[data['EMA_short'] < data['EMA_long'], 'position'] = -1
        data['position'] = data['position'].shift(1)

        # Compute strategy returns
        data['returns'] = data['price'].pct_change()
        data['strategy'] = data['position'] * data['returns']

        # Compute cumulative returns
        data['cumulative_strategy'] = (1 + data['strategy']).cumprod()
        data['cumulative_price'] = (1 + data['returns']).cumprod()

        # Report final values
        strategy_return = data['cumulative_strategy'].iloc[-1]
        buyhold_return = data['cumulative_price'].iloc[-1]

        print(f"Asset {i}: H = {H:.3f}, EMA Strategy = {strategy_return:.2f}, Buy & Hold = {buyhold_return:.2f}")



