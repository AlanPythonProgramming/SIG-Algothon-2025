import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Load data and calculate returns
df = pd.read_csv("prices.txt", sep="\s+", header=None)
returns = df.pct_change().dropna()

# 2. Add 30-day and 100-day EMAs for stock 39
ema_30 = df[39].ewm(span=30, adjust=False).mean()
ema_100 = df[39].ewm(span=100, adjust=False).mean()
ema_30_returns = ema_30.pct_change().dropna()
ema_100_returns = ema_100.pct_change().dropna()

# Align lengths
min_len = min(len(returns), len(ema_30_returns), len(ema_100_returns))
returns = returns.iloc[-min_len:]
ema_30_returns = ema_30_returns.iloc[-min_len:]
ema_100_returns = ema_100_returns.iloc[-min_len:]

# 3. Target and predictors
y = returns[39].shift(-1).iloc[:-1]
X_base = returns.drop(columns=[39]).iloc[:-1]
X_base["EMA_30"] = ema_30_returns.iloc[:-1].values
X_base["EMA_100"] = ema_100_returns.iloc[:-1].values
X_base.columns = X_base.columns.astype(str)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_base, y, shuffle=False, test_size=0.2)

# 4. Forward Stepwise Selection
remaining = list(X_train.columns)
selected = []
best_score = float('inf')
best_pred = None

while remaining:
    scores = []
    for col in remaining:
        trial_features = selected + [col]
        model = LinearRegression().fit(X_train[trial_features], y_train)
        y_pred = model.predict(X_test[trial_features])
        mse = mean_squared_error(y_test, y_pred)
        scores.append((mse, col, y_pred))

    scores.sort()
    best_new_score, best_candidate, candidate_pred = scores[0]

    if best_new_score < best_score:
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_score = best_new_score
        best_pred = candidate_pred
    else:
        break

final_model = LinearRegression().fit(X_train[selected], y_train)
coefficients = final_model.coef_

print("\n📌 Coefficients of Final Selected Predictors:")
for name, coef in zip(selected, coefficients):
    print(f"{name:>15}: {coef:.6f}")

# 5. Directional Accuracy
correct_direction = ((best_pred > 0) == (y_test > 0)).sum()
total = len(y_test)
accuracy = correct_direction / total
print(f"\n✅ Final MSE: {best_score:.6f}")
print(f"📈 Directional Accuracy: {accuracy:.2%} ({correct_direction}/{total})")
print(f"🧠 Selected Predictors: {selected}")

# 6. Optimise threshold for trading
thresholds = np.linspace(0, 0.01, 100)
profits = []
n_trades_list = []
percent_correct_list = []

y_real = y_test.values
best_threshold = 0
best_profit = -np.inf
best_cumulative = None
best_correct = 0
best_trades = 0

for t in thresholds:
    positions = np.where(np.abs(best_pred) > t, np.sign(best_pred), 0)
    daily_profits = positions * 100 * y_real
    total_profit = daily_profits.sum()
    correct_trades = ((positions != 0) & (np.sign(y_real) == positions)).sum()
    n_trades = np.count_nonzero(positions)
    percent_correct = correct_trades / n_trades * 100 if n_trades > 0 else 0

    profits.append(total_profit)
    n_trades_list.append(n_trades)
    percent_correct_list.append(percent_correct)

    if total_profit > best_profit:
        best_profit = total_profit
        best_threshold = t
        best_cumulative = daily_profits.cumsum()
        best_correct = correct_trades
        best_trades = n_trades

# 7. Report and plot results
print(f"\n🚀 Optimal Threshold Found: {best_threshold:.4f}")
print(f"💰 Max Profit: ${best_profit:.2f}")
print(f"📊 Trades Made: {best_trades}, Correct: {best_correct} ({best_correct / best_trades * 100:.2f}%)")

# Profit curve
plt.figure(figsize=(10, 5))
plt.plot(thresholds, profits, label="Total Profit")
plt.axvline(best_threshold, color='red', linestyle='--', label=f"Best Threshold = {best_threshold:.4f}")
plt.title("Profit vs. Threshold")
plt.xlabel("Prediction Threshold")
plt.ylabel("Total Profit ($)")
plt.grid(True)
plt.legend()
plt.show()

# Cumulative profit
plt.figure(figsize=(10, 5))
plt.plot(best_cumulative, label="Cumulative Profit", color='green')
plt.title(f"Cumulative Profit (Optimal Threshold = {best_threshold:.4f})")
plt.xlabel("Test Time Index")
plt.ylabel("Cumulative Profit ($)")
plt.grid(True)
plt.legend()
plt.show()

# Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Returns", alpha=0.7)
plt.plot(best_pred, label="Predicted Returns", alpha=0.7)
plt.title("MLR Prediction of Stock 39 Returns")
plt.xlabel("Test Time Index")
plt.ylabel("Return")
plt.grid(True)
plt.legend()
plt.show()
