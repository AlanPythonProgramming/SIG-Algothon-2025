#!/usr/bin/env python

import numpy as np
import pandas as pd
from main import getMyPosition as getPosition
from matplotlib import pyplot as plt

nInst = 0
nt = 0
commRate = 0.0005
dlrPosLimit = 10000
assets = [2,20,30]
value_array = [0 for x in range(50)]

def loadPrices(fn):
    global nt, nInst
    df=pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt,nInst) = df.shape
    return (df.values).T

pricesFile="./prices.txt"
prcAll = loadPrices(pricesFile)
print ("Loaded %d instruments for %d days" % (nInst, nt))

def calcPL(prcHist, numTestDays): 
    cash = 0
    cash_array = [0 for x in range(50)]
    value_array = [0 for x in range(50)]
    volume_array = []
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0
    todayPLL = []
    todayPLL_array = []
    (_,nt) = prcHist.shape
    startDay = nt + 1 - numTestDays
    for t in range(startDay, nt+1):
        prcHistSoFar = prcHist[:,:t]
        curPrices = prcHistSoFar[:,-1]
        if (t < nt):
            # Trading, do not do it on the very last day of the test
            newPosOrig = getPosition(prcHistSoFar)
            posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
            newPos = np.clip(newPosOrig, -posLimits, posLimits)
            deltaPos = newPos - curPos
            dvolumes = curPrices * np.abs(deltaPos)
            dvolume = np.sum(dvolumes)
            totDVolume += dvolume
            comm = dvolume * commRate
            cash -= curPrices.dot(deltaPos) + comm

            # Calculate pnl for individual assets
            cash_array -= curPrices*deltaPos + dvolumes*commRate

        else:
            newPos = np.array(curPos)
        
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue
        ret = 0.0

        # Array stuff
        posValue_array = curPos*curPrices
        todayPL_array = cash_array + posValue_array - value_array
        value_array = cash_array + posValue_array

        if (totDVolume > 0):
            ret = value / totDVolume
        if (t > startDay):
            print ("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" % (t,value, todayPL, totDVolume, ret))
            todayPLL.append(todayPL)

            todayPLL_array.append(todayPL_array)
            volume_array.append(dvolumes)

    pll = np.array(todayPLL)
    pll_array = pd.DataFrame(todayPLL_array)
    volume_array = pd.DataFrame(volume_array)
    (plmu,plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(249) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume, pll, pll_array, volume_array)

(meanpl, ret, plstd, sharpe, dvol, pll, pll_array, volume_array) = calcPL(prcAll,300)
score = meanpl - 0.1*plstd
print ("=====")
print ("mean(PL): %.1lf" % meanpl)
print ("return: %.5lf" % ret)
print ("StdDev(PL): %.2lf" % plstd)
print ("annSharpe(PL): %.2lf " % sharpe)
print ("totDvolume: %.0lf " % dvol)
print ("Score: %.2lf" % score)


# Calculate and sort assets by profit
profits = np.sum(pll_array,axis = 0)
sortedProfitIndex = np.argsort(-profits)

# Calculate returns for each asset
volumes = np.sum(volume_array,axis = 0)
returns = profits / volumes

for i in range(50):
    print('Asset',sortedProfitIndex[i],": $",
          round(profits[sortedProfitIndex[i]],2),":",
          round(100*returns[sortedProfitIndex[i]],3),"%")




plt.plot(np.cumsum(pll), label='Cumulative P&L')
plt.title("Total Portfolio P&L")
plt.show()

# Filter for just the assets in focus
pll_array = pll_array[assets]

plt.plot(np.cumsum(pll_array,axis=0))
plt.legend(pll_array)
plt.title("Select Assets P&L")
plt.show()

plt.plot(np.cumsum(np.sum(pll_array,axis=1),axis=0))
plt.title("Select Assets Portfolio P&L")
plt.show()