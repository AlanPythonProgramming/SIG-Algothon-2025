import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import json

# 1. Load data
df = pd.read_csv("../prices.txt", delim_whitespace=True, header=None)
returns = df.pct_change().dropna()

results = []

# 2. Loop through each stock as the prediction target
for target_idx in range(50):
    print(f"\n🔄 Target Stock: {target_idx}")
    
    # Prepare target and features
    y = returns[target_idx].shift(-1).iloc[:-1]
    X_full = returns.drop(columns=[target_idx]).iloc[:-1]
    
    X_train, X_test, y_train, y_test = train_test_split(X_full, y, shuffle=False, test_size=0.2)

    # 3. Forward stepwise selection
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

    # 4. Evaluate directional accuracy
    y_test = y_test.reset_index(drop=True)
    correct_direction = ((best_pred > 0) == (y_test > 0)).sum()
    total_predictions = len(y_test)
    directional_accuracy = correct_direction / total_predictions

    results.append({
        "Target Stock": target_idx,
        "Directional Accuracy": directional_accuracy,
        "Correct Predictions": correct_direction,
        "Total Predictions": total_predictions,
        "MSE": best_score,
        "Selected Predictors": selected
    })

# 5. Results DataFrame sorted by accuracy
results_df = pd.DataFrame(results).sort_values(by="Directional Accuracy", ascending=False).reset_index(drop=True)

print("\n✅ Top Predictable Stocks by Directional Accuracy:")
print(results_df[["Target Stock", "Directional Accuracy"]].head(10))

# 6. Plot actual vs predicted for best stock
best_stock = results_df.loc[0, "Target Stock"]
best_features = results_df.loc[0, "Selected Predictors"]

# Prepare data again for best stock
y = returns[best_stock].shift(-1).iloc[:-1]
X_full = returns.drop(columns=[best_stock]).iloc[:-1]
X_train, X_test, y_train, y_test = train_test_split(X_full, y, shuffle=False, test_size=0.2)

# Train model
final_model = LinearRegression().fit(X_train[best_features], y_train)
y_pred_final = final_model.predict(X_test[best_features])

# Plot actual vs predicted returns
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Returns", alpha=0.7)
plt.plot(y_pred_final, label="Predicted Returns", alpha=0.7)
plt.title(f"📈 Actual vs Predicted Returns for Stock {best_stock}")
plt.xlabel("Test Time Index")
plt.ylabel("Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()