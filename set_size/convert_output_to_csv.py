import sys
from collections import defaultdict

# header
sys.stdin.readline()

per_list_repeat_surps = defaultdict(list)

# header
sys.stdin.readline()

print("setSize,surpRatio")
# aggregate surp_ratios over positions in each list
for l in sys.stdin:
    items = l.strip().split()
    names = items[0]
    set_size = int(items[1])
    # retention interval
    surp_ratio = float(items[5])
    per_list_repeat_surps[(names, set_size)].append(surp_ratio)

# print each list's average surprisal ratio
for key, surp_ratios in per_list_repeat_surps.items():
    _, set_size = key
    avg_surp_ratio = sum(surp_ratios) / len(surp_ratios)
    print("{},{}".format(set_size, avg_surp_ratio))
