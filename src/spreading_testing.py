import datetime as dt
import numpy as np
import pandas as pd
from goes2go import GOES
from matplotlib import pyplot as plt
import time

import pirep as pr
import satellite as st
from consts import MAP_RANGE, BACKGROUND_RISK
from pirep.defs.spreading import concatenate_all_pireps, spread_pirep


# Trying to make this code faster.

def main():
    ts_begin = time.time()
    reports = pr.parse_all(pr.fetch(pr.url(dt.datetime(2024, 11, 6, 0, 0, 0, tzinfo=dt.UTC), dt.datetime(2024, 11, 7, 0, 0, 0, tzinfo=dt.UTC))))
    print(f"Number of reports in specified range is {len(reports)}")
    ts_fetch = time.time()
    print(f"Time taken to fetch = {ts_fetch - ts_begin}")

    grid = concatenate_all_pireps(reports, BACKGROUND_RISK)
    ts_concat = time.time()
    print(f"Time taken to concatenate = {ts_concat - ts_fetch}")

    print(f"total time = {ts_concat - ts_begin}")
    
    # Plot a single slice (e.g., the middle slice along the depth axis)
    plt.imshow(grid[:,:, 10], cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title("Slice of Alt Level 10")
    plt.savefig("/skyblue/new_spread.png")

if __name__ == "__main__":
    main()