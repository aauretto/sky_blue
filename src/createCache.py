import pandas as pd
import pirep as pr
import datetime as dt
from pirep.defs.aircraft import Aircraft
from pirep.defs.spreading import spread_pirep
import numpy as np

### STEPS ###
# 1) Read all pireps in date range
# 2) Spread those pireps
# 3) Append them to cache.csv


CSV_FANME = "/skyblue/test_cache.csv"

def create_cache(start, end):
    reports = pr.parse_all(pr.fetch(pr.url(start, end)))

    # Need timestamps and spread grids

    print(type(reports[0]["Timestamp"]))

    # Rip out timestamps for each report
    timestamps = list(map(lambda r : r["Timestamp"], reports))

    pirepLocs = []

    for report in reports:
        grid, aircraft, intensity = pr.compute_grid(report)

        if (intensity, aircraft) == ("NEG", Aircraft.LGT):
            pirepLocs.append((None, report))
        else:
            spread_pirep(grid, aircraft, intensity, 0)
            idxs = np.where(~np.isnan(grid))
            vals = grid[idxs]

            pirepLocs.append((idxs, vals))

    # Convert to pd
    df = pd.DataFrame({
        "Timestamp" : timestamps,
        "Data"      : pirepLocs         
    })
    df.to_csv(CSV_FANME, mode = "a", header=False, index=False)

if __name__ == "__main__":
    start = dt.datetime(2024, 11, 6, 23, 30, 0) 
    end   = dt.datetime(2024, 11, 7,  0,  0, 0) 
    
    create_cache(start, end)