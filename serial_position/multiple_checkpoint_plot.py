import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

sns.set(style="whitegrid")

files = [
    "/users/PAS2157/ceclark/git/free-recall/serial_position/outputs/csv/PYTHIA_1B_CP1_PPL20",
    "/users/PAS2157/ceclark/git/free-recall/serial_position/outputs/csv/PYTHIA_1B_CP4_PPL20",
    "/users/PAS2157/ceclark/git/free-recall/serial_position/outputs/csv/PYTHIA_1B_CP16_PPL20",
    "/users/PAS2157/ceclark/git/free-recall/serial_position/outputs/csv/PYTHIA_1B_CP64_PPL20",
    "/users/PAS2157/ceclark/git/free-recall/serial_position/outputs/csv/PYTHIA_1B_CP256_PPL20",
    "/users/PAS2157/ceclark/git/free-recall/serial_position/outputs/csv/PYTHIA_1B_CP1000_PPL20",
    "/users/PAS2157/ceclark/git/free-recall/serial_position/outputs/csv/PYTHIA_1B_CP4000_PPL20",
    "/users/PAS2157/ceclark/git/free-recall/serial_position/outputs/csv/PYTHIA_1B_CP16000_PPL20",
    "/users/PAS2157/ceclark/git/free-recall/serial_position/outputs/csv/PYTHIA_1B_CP64000_PPL20",
    "/users/PAS2157/ceclark/git/free-recall/serial_position/outputs/csv/PYTHIA_1B_CP143000_PPL20",
]

cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, 10)]


#lms = ["1K", "2K", "3K", "4K", "5K", "6K", "7K", "8K", "9K", "10K"]
lms = ["1", "4", "16", "64", "256", "1K", "4K", "16K", "64K", "143K"]

datasets = list()
for file in files:
    datasets.append(pd.read_csv(file))

fig, ax = plt.subplots()

for j, df in enumerate(datasets):
    sns.lineplot(
        x=df["serialPosition"]+1, y=df["surpRatio"], ax=ax,
        errorbar=('ci', 95), label=lms[j], color=colors[j],
        marker="o"
    )
ax.set_xlabel("Serial Position", fontsize=18)
ax.tick_params(axis='x', labelsize=14)
ax.set_ylabel("Surprisal Ratio", fontsize=18)
ax.tick_params(axis='y', labelsize=14)
ax.legend(fontsize=14)


plt.tight_layout()
plt.savefig(sys.argv[1])
