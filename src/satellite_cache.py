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

BANDS = [8,9,10,13,14,15]
CACHE_DIR = "/skyblue/caching"

def parse_timestamp(ts):
    return ts.strftime("%Y_%m_%d|%H_%M_%S").split("|")

SAT = GOES(satellite=16, product="ABI", domain="C")
def download_images(timestamps):
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
        # timestamp gets wrapped in a numpy object on save so we call .item to unbox it
        return data["timestamp"].item(), data["image"]

    numProcs = len(tsList)
    with ThreadPoolExecutor(max_workers=numProcs) as exec:
        stampsAndImages = exec.map(worker, tsList)

    # Get images and stamps in their own arrays, merge the images
    [allStamps, allImages] = list(zip(*list(stampsAndImages)))
    return (np.vstack(allImages), list(allStamps))

# Code from generator to test against
def sanity_check(timestamps: list[dt.datetime]):
        def worker(ts):
            return st.fetch(ts, SAT)
        
        numProcs = len(timestamps)
        with ThreadPoolExecutor(max_workers=numProcs) as exec:
            xs = exec.map(worker, timestamps)

        # xs = [st.fetch(t, self.sat) for t in timestamps]

        xs = xr.concat(xs, dim="t")
        lats, lons = st.calculate_coordinates(xs)
        xs = st.fetch_bands(xs, BANDS)
        updated_timestamps = [
            dt.datetime.fromtimestamp(
                timestamp.astype("datetime64[s]").astype(int), dt.UTC
            )
            for timestamp in np.array(xs.coords["t"], dtype=np.datetime64)
        ]
        xs = st.smooth(st.project(lats, lons, xs.data))
        return xs, updated_timestamps

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
    ims, times = load_images(tsList)
    end = time.time()

    print(f"Full retreival for images took {end - start}s")

    print("="*100)
    print("Checking against known info:")

    trueData, trueStamps = sanity_check(tsList)
    print(type(trueStamps[0]))
    print(type(times[0]))
    # Check that the array we get from the old method and the array we get from the new method are the same
    # print(trueData == ims)



