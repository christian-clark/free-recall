import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

sns.set(style="whitegrid")

files = [
    "/home/clark.3664/git/cued-recall/serial_position/free_recall/outputs/csv/VANILLA_68512R1_10K_NPL20",
    "/home/clark.3664/git/cued-recall/serial_position/free_recall/outputs/csv/ALIBI_68512R1_10K_NPL20",
    "/home/clark.3664/git/cued-recall/serial_position/free_recall/outputs/csv/FLEETING_68512R1_10K_NPL20",
#    "/home/clark.3664/git/cued-recall/serial_position/free_recall/outputs/csv/FLEETING_68512R2_10K_NPL20",
#    "/home/clark.3664/git/cued-recall/serial_position/free_recall/outputs/csv/FLEETING_68512R3_10K_NPL20"
]

#names = ["run1", "run2", "run3"]
names = ["vanilla", "alibi", "fleeting"]

datasets = list()
for file in files:
    datasets.append(pd.read_csv(file))

fig, ax = plt.subplots()

for j, df in enumerate(datasets):
    sns.lineplot(
        x=df["serialPosition"]+1, y=df["surpRatio"], ax=ax,
        errorbar=('ci', 95), label=names[j], marker="o"
    )
ax.set_xlabel("Serial Position", fontsize=18)
ax.tick_params(axis='x', labelsize=14)
ax.set_ylabel("Surprisal Ratio", fontsize=18)
ax.tick_params(axis='y', labelsize=14)
ax.legend(fontsize=14)

plt.tight_layout()
plt.savefig(sys.argv[1])
