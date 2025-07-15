import numpy as np
import pandas as pd
from main import getMyPosition as getPosition
import matplotlib.pyplot as plt

currentPos = np.zeros(50)
momentum_stocks = []

def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T

def compute_ema(values, span):
    alpha = 2 / (span + 1)
    ema = [values[0]]

    for price in values[1:]:
        ema.append(alpha * price + (1 - alpha) * ema[-1])
    
    return ema

def update_position(inst, prcSoFar, currentPos, short_span, long_span, diff_prop_threshold, multiple):
    price_history = prcSoFar[inst]
    # if len(price_history) < long_span:
    #     return currentPos

    shortMean = compute_ema(price_history[-short_span:], span=short_span)[-1]
    longMean = compute_ema(price_history[-long_span:], span=long_span)[-1]

    diff_prop = (shortMean - longMean) / longMean
    if abs(diff_prop) > diff_prop_threshold:
        currentPos[inst] = multiple * diff_prop

    # currentPos[inst] = 1000 * (shortMean - longMean)

    return currentPos

def getPosition(prcSoFar, n, m, gradient_threshold, short_span, long_span, diff_prop_threshold, multiple):
    global currentPos, momentum_stocks
    currentPos = np.zeros(50)
    if (prcSoFar.shape[1]-1) % m == 0:
        momentum_stocks = gradientInstruments(prcSoFar[:, -n:], gradient_threshold)
    
    for stock in momentum_stocks:
        currentPos = update_position(stock, prcSoFar, currentPos, short_span, long_span, diff_prop_threshold, multiple)
    
    return currentPos

def gradientInstruments(prcs, threshold):
    mean_grads = []
    insts = []
    for inst in range(50):
        gradients = []
        for i in range(len(prcs[inst]) - 1):
            gradient = prcs[inst][i + 1] - prcs[inst][i]
            gradients.append(gradient)
        mean_grads.append(np.mean(gradients))
    
    for inst in range(50):
        if abs(mean_grads[inst]) > threshold:
            insts.append(inst)
        
    return insts

def calcScore(prcHist, numTestDays, n, m, gradient_threshold, short_span, long_span, diff_prop_threshold, multiple):
    nInst = 50
    nt = 0
    commRate = 0.0005
    dlrPosLimit = 10000

    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0
    todayPLL = []
    (_,nt) = prcHist.shape
    startDay = nt + 1 - numTestDays
    for t in range(startDay, nt+1):
        prcHistSoFar = prcHist[:,:t]
        curPrices = prcHistSoFar[:,-1]
        if (t < nt):
            # Trading, do not do it on the very last day of the test
            newPosOrig = getPosition(np.array(prcHistSoFar), n, m, gradient_threshold, short_span, long_span, diff_prop_threshold, multiple)
            posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
            newPos = np.clip(newPosOrig, -posLimits, posLimits)
            deltaPos = newPos - curPos
            dvolumes = curPrices * np.abs(deltaPos)
            dvolume = np.sum(dvolumes)
            totDVolume += dvolume
            comm = dvolume * commRate
            cash -= curPrices.dot(deltaPos) + comm
        else:
            newPos = np.array(curPos)
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue
        ret = 0.0
        if (totDVolume > 0):
            ret = value / totDVolume
        if (t > startDay):
            print ("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" % (t,value, todayPL, totDVolume, ret))
            todayPLL.append(todayPL)
    pll = np.array(todayPLL)
    (plmu,plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(249) * plmu / plstd
    
    score = plmu - 0.1 * plstd
    return score


pricesFile="./priceSlice_test.txt"
prcAll = loadPrices(pricesFile)

# --- Bayesian Optimization Setup ---
from bayes_opt import BayesianOptimization

pbounds = {
    'n_proxy': (0, 4),                 # Lookback period for gradient
    'm_proxy': (0, 1),                   # How often to re-calculate momentum stocks
    'gradient_threshold': (0.05, 0.2),
    'short_span': (5, 50),          # Short EMA window
    'long_span': (50, 250),         # Long EMA window
    'diff_prop_threshold': (0.001, 0.15), # Signal trigger threshold
    'multiple': (500, 5000)         # Position sizing multiplier
}

# 2. Create the objective function for the optimizer
# The optimizer will pass keyword arguments to this function.
def objective_function(n_proxy, m_proxy, gradient_threshold, short_span, long_span, diff_prop_threshold, multiple):
    """
    Wrapper for calcScore that handles parameter types and constraints.
    """
    # --- Map proxy values to your discrete choices ---
    n_values = [50, 100, 150, 200]
    n = n_values[min(int(n_proxy), len(n_values) - 1)] # Map n_proxy to an index

    m = 1 if m_proxy < 0.5 else 100 # Map m_proxy to 1 or 100


    # Ensure integer parameters are integers
    # n = int(n)
    # m = int(m)
    short_span = int(short_span)
    long_span = int(long_span)

    # Add a constraint: short_span must be smaller than long_span.
    # If not, return a very low score to penalize this choice.
    if short_span >= long_span:
        return -1e9  # Negative infinity

    # Run the backtest with the given parameters
    score = calcScore(
        prcHist=prcAll[:, :750],
        numTestDays=750-n, # Using a fixed number of test days
        n=n,
        m=m,
        gradient_threshold=gradient_threshold,
        short_span=short_span,
        long_span=long_span,
        diff_prop_threshold=diff_prop_threshold,
        multiple=multiple
    )
    return score

# 3. Set up and run the Bayesian Optimizer
if __name__ == "__main__":
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42,  # for reproducibility
        verbose=2 # 2 prints all steps, 1 prints only when a maximum is observed
    )

    # Run the optimization.
    # init_points: Number of random exploration steps.
    # n_iter: Number of Bayesian optimization steps.
    optimizer.maximize(
        init_points=5,
        n_iter=1000,
    )

    # Print the best results
    print("\n--- Best Parameters Found ---")
    print(optimizer.max)

# print(calcScore(prcAll[:, :], 100, 100, 100, 0.07, 10, 100, 0.075, 1000))