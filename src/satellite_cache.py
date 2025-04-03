from goes2go import GOES
import datetime as dt
import satellite as st
import multiprocessing
import numpy as np 
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor

import os
import xarray as xr
import pickle

BANDS = [8,9,10,13,14,15]
CACHE_DIR = "/skyblue/caching"

def parse_timestamp(ts):
    return ts.strftime("%Y_%m_%d|%H_%M_%S").split("|")

def download_images(timestamps):
    SAT = GOES(satellite=16, product="ABI", domain="C")
    def worker(ts):
        startTime = time.time()
        yyyymmdd, hhmmss = parse_timestamp(ts)
        thisDir = f"{CACHE_DIR}/{yyyymmdd}"
        outfile = f"{thisDir}/{hhmmss}.npz"

        rawXarr = st.fetch(ts, SAT)
        lats, lons = st.calculate_coordinates(rawXarr)

        wTime = xr.concat([rawXarr], dim="t")
        updated_timestamp = [ # we only want one timestamp here
            dt.datetime.fromtimestamp(
                timestamp.astype("datetime64[s]").astype(int), dt.UTC
            )
            for timestamp in np.array(wTime.coords["t"], dtype=np.datetime64)
        ][0]
        trimmed = st.fetch_bands(wTime, BANDS)
        smoothed = st.smooth(st.project(lats, lons, trimmed.data))
        os.makedirs(thisDir, exist_ok=True)
        np.savez(outfile, image=smoothed, timestamp=updated_timestamp)
        endTime = time.time()
        print(f"Saved to {outfile} [{endTime-startTime:.4f}s]")
 

    numProcs = len(timestamps)
    with ThreadPoolExecutor(max_workers=numProcs) as exec:
        xs = exec.map(worker, timestamps)
    

def load_images(tsList):
    
    def worker(ts):
        startTime = time.time()
        yyyymmdd, hhmmss = parse_timestamp(ts)
        infile = f"{CACHE_DIR}/{yyyymmdd}/{hhmmss}.npz"
        data = np.load(infile, allow_pickle = True)
        endTime = time.time()

        print(f"Loaded from {infile} [{endTime-startTime:.4f}s]")

        return data["timestamp"], data["image"]

    numProcs = len(tsList)
    with ThreadPoolExecutor(max_workers=numProcs) as exec:
        stampsAndImages = exec.map(worker, tsList)

    # NOTE vstack the arrays
    return(list(stampsAndImages))

if __name__ =='__main__':

    print("CPU count (multiprocessing):", multiprocessing.cpu_count())

    multiprocessing.set_start_method("forkserver", force=True)

    ts = dt.datetime(2020, 1, 1, 0, 0)
    delts = dt.timedelta(hours = 1)
    tsList = [ts + delts * i for i in range(9)]
    print(*tsList, sep = "\n")
    
    start = time.time()
    download_images(tsList)
    end = time.time()
    print(f"Full caching for 9 procs took {end - start}s")

    start = time.time()
    ims = load_images(tsList)
    end = time.time()

    print(f"Full retreival for images took {end - start}s")

    print(ims[0][0])
    print(ims[0][1])

