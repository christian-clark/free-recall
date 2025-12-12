import sys

# header
sys.stdin.readline()

print("serialPosition,surpRatio")
for l in sys.stdin:
    items = l.strip().split()
    serial_position = int(items[1])
    surp_ratio = float(items[4])
    print("{},{}".format(serial_position, surp_ratio))
