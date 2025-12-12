import seaborn as sns
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import sys

df = pd.read_csv(sys.argv[1])

sns.set_theme(style="whitegrid")
ax = sns.lineplot(data=df, x="setSize", y="surpRatio", errorbar=('ci', 95),  color="black")

ax.set_xlabel("Set Size", fontsize=14)
ax.set_ylabel("Surprisal Ratio", fontsize=14)
ax.tick_params(axis="both", labelsize=12)

plt.tight_layout()
plt.savefig(sys.argv[2])

