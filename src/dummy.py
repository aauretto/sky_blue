from goes2go import GOES
import datetime as dt
import satellite as st
import multiprocessing
import numpy as np 
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor

import xarray as xr

SAT = GOES(satellite=16, product="ABI", domain="C")


def worker(ts):
    startTime = time.time()
    r = st.fetch(ts, SAT)
    endTime = time.time()
    print(f"Proc'd file in {endTime - startTime}s")
    return r

def launch_procs_mp(tsList):
    numProcs = len(tsList)
    with multiprocessing.Pool(processes=numProcs) as pool:  # Use an appropriate number of processes
        results = pool.map(worker, tsList)  # Submit 9 copies with different arguments
    return results

def launch_procs_threaded(tsList):
    numProcs = len(tsList)
    # with multiprocessing.Pool(processes=numProcs) as pool:  # Use an appropriate number of processes
    #     results = pool.map(worker, tsList)  # Submit 9 copies with different arguments
    with ThreadPoolExecutor(max_workers=numProcs) as exec:
        results = exec.map(worker, tsList)


    return results

def single_read(tsList):
    results = map(worker, tsList)
    return list(results)


BANDS = [8,9,10,13,14,15]

def download_from_list(timestamps):
    def worker(ts):
            return st.fetch(ts, SAT)
        
    numProcs = len(timestamps)
    with ThreadPoolExecutor(max_workers=numProcs) as exec:
        xs = exec.map(worker, timestamps)
    xs = xr.concat(xs, dim="t")
    # xs = [st.fetch(t, self.sat) for t in timestamps]
    lats, lons = st.calculate_coordinates(xs)
    xs = st.fetch_bands(xs, BANDS)
    updated_timestamps = [
        dt.datetime.fromtimestamp(
            timestamp.astype("datetime64[s]").astype(int), dt.UTC
        )
        for timestamp in np.array(xs.coords["t"], dtype=np.datetime64)
    ]
    xs = st.smooth(st.project(lats, lons, xs.data))

    np.save("/skyblue/caching/testFile1.nc", xs)
    
    assert xs.shape[1:] == (
        1500,
        2500,
        len(BANDS),
    )




if __name__ =='__main__':

    print("CPU count (multiprocessing):", multiprocessing.cpu_count())

    multiprocessing.set_start_method("forkserver", force=True)

    ts = dt.datetime(2020, 1, 1, 0, 0)
    delts = dt.timedelta(hours = 1)
    tsList = [ts + delts * i for i in range(9)]
    print(*tsList, sep = "\n")
    
    

    start = time.time()

    download_from_list(tsList)

    end = time.time()

    print(f"Full caching for 9 procs took {end - start}s")

