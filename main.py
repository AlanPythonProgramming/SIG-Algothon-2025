
import numpy as np
from sklearn.linear_model import LinearRegression

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)

# MVO IMPLEMENTATION

# def weights_to_position(weights, curPrices, cap_per_asset=10000):
#     max_w = np.max(np.abs(weights))
#     if max_w == 0:
#         return np.zeros_like(weights)
    
#     scale = cap_per_asset / max_w
#     dollar_positions = weights * scale
#     positions = np.floor(dollar_positions / curPrices).astype(int)
#     return positions

# def getMyPosition(prcSoFar): # MVO using purely historical estimates of return and covariance
#     global currentPos
#     (nins, nt) = prcSoFar.shape
#     if nt < 21:
#         return np.zeros(nins)
    
#     returns = np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1])
#     cov = np.cov(returns)
#     mu = np.mean(returns, axis=1)

#     inv_cov = np.linalg.pinv(cov)
#     weights = inv_cov @ mu / np.abs(np.sum(inv_cov @ mu))
#     currentPos = weights_to_position(weights, prcSoFar[:, -1])
#     return currentPos

# MVO IMPLEMENTATION WITH SMOOTHING AND LEDOIT-WOLF ESTIMATION
# from sklearn.covariance import LedoitWolf

# nInst = 50
# lookback = 20          # 20 日收益估计 μ、Σ
# k = 6                  # 只交易最强的 6 只（3 long 3 short）
# week_rebal = 10        # 5 个交易日调仓一次
# decay = 0.4            # 旧仓位保留 30 %
# cap_per_asset = 1000

# # 全局记录
# last_pos = np.zeros(nInst, dtype=int)

# def weights2shares(w, price):
#     """ 把权重 w 转成 share；最大单边 10k """
#     max_w = np.max(np.abs(w))
#     if max_w == 0:
#         return np.zeros_like(w, dtype=int)
#     scale = cap_per_asset / max_w
#     shares = np.floor((w * scale) / price).astype(int)
#     cap = np.floor(cap_per_asset / price).astype(int)
#     return np.clip(shares, -cap, cap)

# def getMyPosition(prcSoFar):
#     global last_pos
#     n, t = prcSoFar.shape
#     today = t - 1

#     # ---------- 1. 只在星期五 (或第 5,10,… 天) 调仓 ----------
#     if today % week_rebal != 0 or t < lookback + 1:
#         return last_pos

#     # ---------- 2. 计算 20 日对数收益 ----------
#     ret = np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1])
#     win = ret[:, -lookback:]                 # rolling window
#     mu = win.mean(axis=1)
#     Sigma = LedoitWolf().fit(win.T).covariance_

#     # ---------- 3. Σ⁻¹ μ 方向，但只取最极端的 k 只 ----------
#     raw = np.linalg.solve(Sigma + 1e-6*np.eye(nInst), mu)
#     sel = np.argsort(np.abs(raw))[-k:]       # top-|raw|
#     w = np.zeros(nInst)
#     w[sel] = raw[sel]

#     # 净暴露调零 & L1 归一化 (∑|w|=1)
#     w -= w.mean()
#     w /= np.sum(np.abs(w)) + 1e-12

#     # ---------- 4. 转成 share，并做衰减 ----------
#     price = prcSoFar[:, -1]
#     target_shares = weights2shares(w, price)
#     new_pos = np.round(decay * last_pos + (1 - decay) * target_shares).astype(int)

#     last_pos = new_pos
#     return new_pos

# PAIRS TRADING IMPLEMENTATION

# cointegrated_pairs = np.load('train_successful_coints.npy')
# flat_indices = np.argsort(cointegrated_pairs, axis=None)[::-1]
# indices = list(zip(*np.unravel_index(flat_indices, cointegrated_pairs.shape)))
# top_pairs = indices[:10]

# dollar_limit = 10000
# lookback = 60  # window for beta/spread/zscore
# entry_z = 3
# exit_z = 1.5

# last_pair_positions = {}

# def calc_beta(y, x):
#     """Estimate hedge ratio (beta) using linear regression."""
#     model = LinearRegression().fit(x.reshape(-1, 1), y)
#     return model.coef_[0]

# def get_spread(y, x, beta):
#     return y - beta * x

# def get_zscore(spread):
#     mean = np.mean(spread)
#     std = np.std(spread)
#     return (spread[-1] - mean) / (std + 1e-8)

# def getMyPosition(prcSoFar):
#     n, t = prcSoFar.shape
#     pos = np.zeros(n, dtype=int)
#     global last_pair_positions

#     for (i, j) in top_pairs:
#         if t < lookback + 1:
#             continue
#         y = prcSoFar[i, -lookback:]
#         x = prcSoFar[j, -lookback:]
#         beta = calc_beta(y, x)
#         spread = get_spread(y, x, beta)
#         z = get_zscore(spread)

#         # Trading logic
#         prev = last_pair_positions.get((i, j), 0)
#         if z > entry_z:
#             # Short spread: short y, long x
#             pos_y = -dollar_limit // prcSoFar[i, -1]
#             pos_x = int(beta * dollar_limit // prcSoFar[j, -1])
#             pos[i] += pos_y
#             pos[j] += pos_x
#             last_pair_positions[(i, j)] = -1
#         elif z < -entry_z:
#             # Long spread: long y, short x
#             pos_y = dollar_limit // prcSoFar[i, -1]
#             pos_x = -int(beta * dollar_limit // prcSoFar[j, -1])
#             pos[i] += pos_y
#             pos[j] += pos_x
#             last_pair_positions[(i, j)] = 1
#         elif abs(z) < exit_z:
#             # Exit position
#             last_pair_positions[(i, j)] = 0
#             # No position for this pair

#     # Clip per-instrument positions to dollar limit
#     for idx in range(n):
#         max_shares = dollar_limit // prcSoFar[idx, -1]
#         pos[idx] = np.clip(pos[idx], -max_shares, max_shares)

#     return pos

# def getMyPosition(prcSoFar):
#     returns = np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1])
#     mean_returns = np.mean(returns, axis=1)
#     k = 5
#     top_k = np.argsort(mean_returns)[-k:]
#     bottom_k = np.argsort(mean_returns)[:k]
#     pos = np.zeros(nInst, dtype=int)
#     for i in top_k:
#         pos[i] = -10000 // prcSoFar[i, -1]
#     for i in bottom_k:
#         pos[i] = 10000 // prcSoFar[i, -1]
#     return pos

import numpy as np

nInst = 50
dollar_limit = 10000
confidence_z = 2.0  # Threshold for "super confident" signal

def kalman_filter(prices, Q=0.0001, R=0.01):
    """Simple 1D Kalman filter for price series."""
    n = len(prices)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhat[0] = prices[0]
    P[0] = 1.0
    for t in range(1, n):
        # Prediction
        xhatminus = xhat[t-1]
        Pminus = P[t-1] + Q
        # Update
        K = Pminus / (Pminus + R)
        xhat[t] = xhatminus + K * (prices[t] - xhatminus)
        P[t] = (1 - K) * Pminus
    return xhat

def getMyPosition(prcSoFar):
    n, t = prcSoFar.shape
    pos = np.zeros(n, dtype=int)
    lookback = 50  # Use last 50 days for confidence estimation

    for i in range(n):
        prices = prcSoFar[i]
        if len(prices) < lookback + 2:
            continue
        # Run Kalman filter on this instrument
        kf_est = kalman_filter(prices)
        # Calculate residuals (prediction errors)
        residuals = prices[-lookback:] - kf_est[-lookback:]
        mean_resid = np.mean(residuals)
        std_resid = np.std(residuals) + 1e-8
        zscore = (residuals[-1] - mean_resid) / std_resid

        # Trading logic: act only if "super confident"
        if zscore > confidence_z:
            # Price much higher than trend: short
            pos[i] = -dollar_limit // prices[-1]
        elif zscore < -confidence_z:
            # Price much lower than trend: long
            pos[i] = dollar_limit // prices[-1]
        else:
            pos[i] = 0
    return pos