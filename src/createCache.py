import pandas as pd
import pirep as pr
import datetime as dt
from pirep.defs.aircraft import Aircraft
from pirep.defs.spreading import spread_pirep
import numpy as np
import sys

### STEPS ###
# 1) Read all pireps in date range
# 2) Spread those pireps
# 3) Append them to cache.csv


CSV_FANME = "/skyblue/test_cache.csv"

def create_cache(start, end):
    reports = pr.parse_all(pr.fetch(pr.url(start, end)))

    # Need timestamps and spread grids

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

# parses all nones into actual stuff 
def parse_negs():
    ...

def pull_from_cache(start, end):
    # Reports has: (idx, val) OR (None, report)
    reports = []

    for chunk in pd.read_csv(CSV_FANME, chunksize=10):
        # Look at last ts and if its before start continue
        if pd.to_datetime(chunk['Timestamp'].iloc[len(chunk) - 1]).to_pydatetime() < start:
            print("CONTINUING, last elem < Start")
            continue
        
        # Look at first ts and if its after end break
        if pd.to_datetime(chunk['Timestamp'].iloc[0]).to_pydatetime() > end:
            print("BREAKING, first elem > END")
            break
        
        # Read entire chunk if it is in range:
        if pd.to_datetime(chunk['Timestamp'].iloc[0]).to_pydatetime() > start and \
           pd.to_datetime(chunk['Timestamp'].iloc[len(chunk) - 1]).to_pydatetime() < end:
            reports += chunk['Data'].tolist()
        
        else: # find latest endpoint before end
            start_idx = None
            end_idx   = len(chunk)
            for index, row in chunk.iterrows():
                # TODO: ask Tanay if we are incl or excl on the pirep time ranges
                if pd.to_datetime(row['Timestamp']).to_pydatetime() > start and start_idx is None:
                    start_idx = index
                if pd.to_datetime(row['Timestamp']).to_pydatetime() > end:
                    print("in end case: ", pd.to_datetime(row['Timestamp']).to_pydatetime())
                    end_idx   = index
                    break
            print(f"slicing: [{start_idx}:{end_idx}]")
            reports += chunk['Data'].iloc[start_idx:end_idx].tolist()
        
    return reports
        


if __name__ == "__main__":
    if '--load' not in sys.argv:
        # Intialize csv for appending later
        if '--init' in sys.argv:
            df = pd.DataFrame({
                "Timestamp" : [],
                "Data"      : []
            })
            df.to_csv(CSV_FANME, mode = "w", header=True, index=False)
        else:
            for hr in range(5):
                start = dt.datetime(2024, 11, 7,  0,  0, 0) 
                end   = dt.datetime(2024, 11, 7,  0, 30, 0) 
                diff = dt.timedelta(hours=hr)
                create_cache(start + diff, end + diff)

            df = pd.read_csv(CSV_FANME)
    else:
        start = dt.datetime(2024, 11, 7, 0, 2)
        end = dt.datetime(2024, 11, 7, 0, 5)
        reports = pull_from_cache(start, end)
        df = pd.DataFrame({'Data': reports})
        df.to_csv('/skyblue/res.csv')

    