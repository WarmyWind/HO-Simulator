import sys
import time

def progress_bar(pct):
    print("\r", end="")
    print("Simulation Progress: {:.2f}%: ".format(pct), "▋" * int(pct // 2), end="")
    if pct != 100:
        sys.stdout.flush()
    else:
        print('\n')