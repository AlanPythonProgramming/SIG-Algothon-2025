import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("prices.txt", delim_whitespace=True, header=None)


def gradientFinder(bus):
    gradients = []
    for i in range(len(bus) - 1):
        gradient = bus.iloc[i + 1] - bus.iloc[i]
        gradients.append(gradient)
    return np.mean(gradients)

df = df[499:749]

bus = []
for i in range(50):
    if (abs(gradientFinder(df[i])) > 0.05):
        # plt.plot(df[i])
        # plt.show()
        bus.append(i)
print(bus)

