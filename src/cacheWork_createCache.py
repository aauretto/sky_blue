"""
File: createCache.py
Purpose: 
    Creates a pickled dataset of PIREPs for later use in model training
    See constants and main for usage

Usage:
    run two commands for correct usage:
    python createCache.py --init
    python createCache.py
"""

import sys
import pandas as pd
import pirep as pr
import datetime as dt
import numpy as np
import pickle

# Constants: file path of the CSV and the error log file
CSV_FANME = "/skyblue/testcache.csv"
ERR_LOG_FILE = '/skyblue/testcacheErrorLog.txt'

def create_cache(start, end):
    """
    Creates a pickled cache of pireps from start to end with two columns
    'Timestamps' and 'Data' where Timestamps are the reports' tiemstamps,
    and 'Data' is the pickled reports

    Parameters
    ----------
    start: dt.datetime
        The start date (in UTC) to include PIREPs from
    end: dt.datetime
        The end data (in UTC) to include PIREPs to
    """
    print(f"Creating New Batch for {start} to {end}")
    reports = pr.parse_all(pr.fetch(pr.url(start, end)))

    # Rip out timestamps for each report
    timestamps = list(map(lambda r : r["Timestamp"], reports))


    # Convert to pd
    df = pd.DataFrame({
        "Timestamp" : timestamps,
        "Data"      : [pickle.dumps(r) for r in reports]     
    })
    
    df.to_csv(CSV_FANME, mode = "a", header=False, index=False)


if __name__ == "__main__":

    if '--init' in sys.argv:
        df = pd.DataFrame({
            "Timestamp" : [],
            "Data"      : []
        })
        df.to_csv(CSV_FANME, mode = "w", header=True, index=False)
    else:
        # Start is the start date to pull PIREPs from and final is the last data to pull PIREPs up to
        # start = dt.datetime(2017, 4, 5, 12,  0, 0, tzinfo=dt.UTC)
        # final = dt.datetime(2025, 1, 1, 0, 20, 0, tzinfo=dt.UTC)
        # diff  = dt.timedelta(hours=12)
        # end   = start + diff
        start = dt.datetime(2017, 1, 1, 0,  0, 0, tzinfo=dt.UTC)
        final = dt.datetime(2017, 1, 1, 12, 20, 0, tzinfo=dt.UTC)
        diff  = dt.timedelta(hours=12)
        end   = start + diff

        # Will log failed batches for use in cacheFixes.py
        with open(ERR_LOG_FILE, 'a') as errFile:
            while end < final:
                try:
                    create_cache(start, end)
                except Exception as e:
                    print(f"Issue creating cache on range: {start} - {end} with Error:\n {e}", file=errFile)
                diff = dt.timedelta(hours=12)
                start = end + dt.timedelta(milliseconds=1)
                end += diff

            try:
                create_cache(start, final)
            except Exception as e:
                print(f"Issue creating cache on range: {start} - {final} with Error:\n {e}", file=errFile)
                



    