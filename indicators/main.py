import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

def compute_ema(values, span):
    alpha = 2 / (span + 1)
    ema = [values[0]]

    for price in values[1:]:
        ema.append(alpha * price + (1 - alpha) * ema[-1])
    
    return ema

def update_position(inst, prcSoFar, currentPos, long_span):
    price_history = prcSoFar[inst]
    if len(price_history) < long_span:
        return currentPos

    shortMean = compute_ema(price_history[-10:], span=10)[-1]
    longMean = compute_ema(price_history[-long_span:], span=long_span)[-1]


    diff = shortMean - longMean

    currentPos[inst] = 1000 * diff
    return currentPos

def getMyPosition(prcSoFar):
    global currentPos
    currentPos = np.zeros(50)
    
    momentum_stocks = [2,3,4,7,9,16,17,22,23,28,31,32,34,36,41,43,46]
    for stock in momentum_stocks:
        currentPos = update_position(stock, prcSoFar, currentPos, long_span=100)

    if len(prcSoFar[0]) == 999:
        for stock in momentum_stocks:
            plt.plot(prcSoFar[stock], label=f"Stock {stock}")
        plt.title("Momentum Stocks at Final Day")
        plt.legend()
        plt.show()

    return currentPos
