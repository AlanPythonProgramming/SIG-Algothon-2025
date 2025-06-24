import numpy as np
import pandas as pd
import plotly.express as px

from xgboost import XGBRegressor, XGBClassifier, plot_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from matplotlib import pyplot as plt
import joblib

def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T

pricesFile="prices.txt"
prices = loadPrices(pricesFile)
returns = np.diff(np.log(prices), axis=1)
returns_df = pd.DataFrame(returns.T)
prices_df = pd.DataFrame(prices.T)

prices = pd.DataFrame(prices.T)
df = pd.DataFrame(returns.T)
df.columns = df.columns.astype(str)
train = df[:510].reset_index(drop=True)
test = df[500:].reset_index(drop=True)
val_mse = []
ema_ranges = [2, 5, 10, 20, 50, 80]
reg_models = []
important_features = []

def compute_features(rets, prcs, inst, ema_ranges):
    inst = str(inst)
    features = rets.copy()
    for ema in ema_ranges:
        features[f'ema_{ema}'] = rets[inst].ewm(span=ema, adjust=False).mean()
    # for ema in ema_ranges:
    #     features[f'prices_ema_{ema}'] = prcs[inst].ewm(span=ema, adjust=False).mean()
    features['ma_5'] = rets[inst].rolling(window=5).mean()
    features['mom_5'] = rets[inst] / (rets[inst].shift(5) + 1e-6) - 1
    features['vol_10'] = rets[inst].rolling(window=10).std()
    return features

for threshold in [1e-3]:
    val_direction_accs = []
    print(f"---------Threshold------------: {threshold}")
    for inst in range(nInst):
        X_train_inst = compute_features(train, prices[:510], inst, ema_ranges)
        y_train_inst = train.iloc[:, inst].shift(-10).dropna()
        X_train_inst = X_train_inst[9:-1].reset_index(drop=True)  # Align lengths after shifting
        mses = []
        direction_accs = []
        for fold in range(5):
            fold_size = len(X_train_inst) // 5
            start = fold * fold_size
            end = start + fold_size
            X_val = X_train_inst.iloc[start:end]
            y_val = y_train_inst.iloc[start:end]
            X_train = X_train_inst.drop(X_val.index)
            y_train = y_train_inst.drop(y_val.index)
            
            reg_model = LinearRegression()

            reg_model.fit(X_train, y_train)
            y_pred = reg_model.predict(X_val)

            y_pred_sign = np.full(len(y_val), -1)
            high_prob_1 = y_pred > threshold
            high_prob_0 = y_pred < -threshold
            y_pred_sign[high_prob_1] = 1
            y_pred_sign[high_prob_0] = 0

            # Only evaluate accuracy where a confident prediction was made
            mask = y_pred_sign != -1
            if np.any(mask):
                acc = accuracy_score((y_val[mask] > 0).astype(int), y_pred_sign[mask])
                direction_accs.append((np.sum(mask), acc))
            else:
                direction_accs.append((0, 0))
        
        total_confident = sum(x[0] for x in direction_accs)
        mean_acc = np.mean([x[1] for x in direction_accs if x[0] > 0])
        val_direction_accs.append((total_confident, mean_acc))
        print(f"Instrument {inst}: Total Confident Predictions = {total_confident}, Mean Accuracy = {mean_acc:.4f}")

        # model = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
        # model.fit(X_train_inst, y_train_inst)
        # reg_models.append(model)

    # Find top 10 instruments with highest mean accuracy for this threshold
    val_direction_accs_arr = np.array(val_direction_accs, dtype=object)
    mean_accs = np.array([x[1] for x in val_direction_accs])
    top10_idx = np.argsort(mean_accs)[-10:][::-1]
    print(f"Top 10 instruments for threshold {threshold}:")
    for rank, idx in enumerate(top10_idx, 1):
        print(f"{rank}. Instrument {idx}: Mean Accuracy = {mean_accs[idx]:.4f}, Confident Predictions = {val_direction_accs[idx][0]}")
    
    overall_confident = sum(x[0] for x in val_direction_accs)
    overall_acc = np.mean([x[1] for x in val_direction_accs if x[0] > 0])
    print(f"Total Confident Predictions and Mean Accuracy: ({overall_confident}, {overall_acc}) for threshold {threshold})")

    top_10_vals = [val_direction_accs[i] for i in top10_idx]
    top_10_confident = sum(x[0] for x in top_10_vals)
    top_10_acc = np.mean([x[1] for x in top_10_vals if x[0] > 0])
    print(f"Top 10 Confident Predictions and Mean Accuracy: ({top_10_confident}, {top_10_acc}) for threshold {threshold})")

# print("Training complete. Saving", len(reg_models), "models.")
# joblib.dump(reg_models, 'xgb_regression_models.joblib')

# df = pd.DataFrame(test_prices.T, columns=[f'Instrument {i}' for i in range(nInst)])
# test = []
# for inst in range(nInst):
#     rets = np.log(df.iloc[:, inst] / df.iloc[:, inst].shift(1)) * 100
#     tmp = pd.DataFrame({
#         # 'inst': inst,
#         # 'price': p,
#         'lret': rets,
#         'ema_3': rets.ewm(span=3, adjust=False).mean(),
#         'ema_8': rets.ewm(span=8, adjust=False).mean(),
#         'ma_5': rets.rolling(window=5).mean(),
#         'mom_5': rets / (rets.shift(5)+1e-6) - 1,
#         'vol_10': rets.rolling(window=10).std(),
#         'fret': rets.shift(-1),
#         'direction': np.sign(rets.shift(-1)),
#     })
#     test.append(tmp)
# test = pd.concat(test).dropna().reset_index(drop=True)