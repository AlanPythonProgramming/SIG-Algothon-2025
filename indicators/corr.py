import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("prices.txt", delim_whitespace=True, header=None)

corr_matrix = df.corr()

# plt.figure(figsize=(10, 8))
# sns.heatmap(
#     corr_matrix,
#     cmap="viridis",       
#     center=0,
#     square=True,
#     cbar_kws={"shrink": 0.75},
#     linewidths=0.3
# )

corr_threshold = 0.9

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        if (i != j):
            if (abs(corr_matrix[i][j]) > corr_threshold):
                print(i, j, corr_matrix[i][j])

bus = [2,4,6,20,22]
for i in bus:
    plt.plot(df[i])
plt.show()


# plt.title("Correlation Heatmap of Raw Prices", fontsize=14)
# plt.xlabel("Assets")
# plt.ylabel("Assets")
# plt.tight_layout()
# plt.show()