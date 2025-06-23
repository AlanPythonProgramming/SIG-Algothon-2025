
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)
models = joblib.load('alan/xgb_classification_models.joblib')  # Load pre-trained models for each stock
ema_ranges = [2, 5, 10, 20, 50, 80]
threshold = 0.95

def compute_features(df, inst, ema_ranges):
    features = pd.DataFrame(index=df.index)
    for ema in ema_ranges:
        features[f'ema_{ema}'] = df[inst].ewm(span=ema, adjust=False).mean()
    features['ma_5'] = df[inst].rolling(window=5).mean()
    features['mom_5'] = df[inst] / (df[inst].shift(5) + 1e-6) - 1
    features['vol_10'] = df[inst].rolling(window=10).std()
    return features

def getMyPosition(prcSoFar):
    global currentPos
    returns = pd.DataFrame(np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1]).T)
    for i in range(nInst):
        X = compute_features(returns, i, ema_ranges).iloc[[-1]]
        # X = returns.iloc[[-1]]
        probs = []
        for j in range(5):
            probs.append(models[i*5+j].predict_proba(X)[0])

        prob = np.mean(probs, axis=0)
        if prob[1] > threshold: 
            currentPos[i] = 1000 / prcSoFar[i, -1] 
        elif prob[0] > threshold: 
            currentPos[i] = -1000 / prcSoFar[i, -1] 

    return currentPos