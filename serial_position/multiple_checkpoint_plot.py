import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

sns.set(style="whitegrid")

files = [
    "/home/clark.3664/git/cued-recall/serial_position/free_recall/outputs/csv/VANILLA_68512R1_1K_PPL20",
    "/home/clark.3664/git/cued-recall/serial_position/free_recall/outputs/csv/VANILLA_68512R1_2K_PPL20",
    "/home/clark.3664/git/cued-recall/serial_position/free_recall/outputs/csv/VANILLA_68512R1_3K_PPL20",
    "/home/clark.3664/git/cued-recall/serial_position/free_recall/outputs/csv/VANILLA_68512R1_4K_PPL20",
    "/home/clark.3664/git/cued-recall/serial_position/free_recall/outputs/csv/VANILLA_68512R1_5K_PPL20",
    "/home/clark.3664/git/cued-recall/serial_position/free_recall/outputs/csv/VANILLA_68512R1_6K_PPL20",
    "/home/clark.3664/git/cued-recall/serial_position/free_recall/outputs/csv/VANILLA_68512R1_7K_PPL20",
    "/home/clark.3664/git/cued-recall/serial_position/free_recall/outputs/csv/VANILLA_68512R1_8K_PPL20",
    "/home/clark.3664/git/cued-recall/serial_position/free_recall/outputs/csv/VANILLA_68512R1_9K_PPL20",
    "/home/clark.3664/git/cued-recall/serial_position/free_recall/outputs/csv/VANILLA_68512R1_10K_NPL20"
]

cmap = plt.get_cmap('viridis')
colors = [cmap(i) for i in np.linspace(0, 1, 10)]


lms = ["1K", "2K", "3K", "4K", "5K", "6K", "7K", "8K", "9K", "10K"]

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
