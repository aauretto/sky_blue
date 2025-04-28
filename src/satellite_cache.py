"""
File satellite_cache.py acts as an in-between for accessing GOES satellite
images. An image can be requested using retrieve_satellite_data. If the 
requested image is saved to disk (cached) already, access should be fast, if
not, it is first saved to disk then accessed so subsequent accesses can be
quicker.  

Usage: Call retreive_satellite_data with a list of timestamps for the images 
       requested. If the user wishes to build a cache, run this file directly
       and the time range specified at the top of the `main` function in this
       file will be cached in the directory specified by CACHE_DIR.

Effects:
    Running code in this file will populate the folder specified by CACHE_DIR
    with an image per timestamp requested. Cached images are large (~90-100MB) 
    per file.
"""

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

### CONSTANTS USED WHEN BUILDING A CACHE:
MAX_PROCESSES = 24
WINDOW_PER_HOUR = 12
BANDS = [8,9,10,13,14,15]
CACHE_DIR = "/cluster/tufts/capstone25skyblue/Caches/sat_cache"
SAT = GOES(satellite=16, product="ABI", domain="C")


def parse_timestamp(ts):
    """
    Parse a datetime.datetime and return a formatted strings

    Parameters
    ----------
    ts: datetime.datetime

    Returns
    -------
    list(str)
        With the format [YYYY_MM_DD, HH_MM_SS]
    """
    return ts.strftime("%Y_%m_%d|%H_%M_%S").split("|")

def cache_worker(ts):
    """
    Downloads the GOES-16 image nearest to the timestamp provided
    Stores the result to a file in CACHE_DIR/yyyy_MM_DD/HH_MM_SS.npz
    
    Parameters
    ----------
    timestamp: datetime.datetime
        The expected tzinfo is UTC
    
    Return
    ------
    None
    """
    print(f"\x1b[38;5;242m[{ts}] > starting worker\x1b[0m")
    startTime = time.time()
    # Figure out where we are going to store the cached file
    yyyymmdd, hhmmss = parse_timestamp(ts)
    thisDir = f"{CACHE_DIR}/{yyyymmdd}"
    outfile = f"{thisDir}/{hhmmss}.npz"

    if os.path.isfile(outfile):
        print(f"\x1b[38;5;242m[{ts}] > {outfile} already exists. Skipping...\x1b[0m")
        return

    # Get data from AWS, compute updated timestamp for data, pull out bands we want
    rawXarr = st.fetch(ts, SAT)
    fetchTime = time.time()
    print(f"\x1b[95m[{ts}] > Fetching took {fetchTime - startTime}s\x1b[0m")

    lats, lons = st.calculate_coordinates(rawXarr)

    # To cooperate with contract for fetch_bands, project, and smooth
    wTime = xr.concat([rawXarr], dim="t")
    updated_timestamp = [ # we only want one timestamp here
        dt.datetime.fromtimestamp(
            timestamp.astype("datetime64[s]").astype(int), dt.UTC
        )
        for timestamp in np.array(wTime.coords["t"], dtype=np.datetime64)
    ][0]
    trimmed = st.fetch_bands(wTime, BANDS)
    smoothed = st.smooth(st.project(lats, lons, trimmed.data))
    procTime = time.time()
    print(f"\x1b[92m[{ts}] > Processing took {procTime - fetchTime}s\x1b[0m")
    
    # Write file to disk
    os.makedirs(thisDir, exist_ok=True)
    np.savez(outfile, image=smoothed, timestamp=updated_timestamp)
    endTime = time.time()
    print(f"\x1b[94m[{ts}] > Saving to disk took {endTime - procTime}s\x1b[0m")
    print(f"[{ts}] > Total time was {endTime - startTime}s")

def cache_images_from_aws(timestamps):
    """
    Downloads the GOES-16 image nearest to each timestamp in the provided
    list of timestamps.
    Stores the results to a files in CACHE_DIR
    
    Parameters
    ----------
    timestamps: list(datetime.datetime)
        The expected tzinfo is UTC
    
    Return
    ------
    None
    """
    # Launch job to download all timestamps over multiple threads
    with multiprocessing.Pool(processes=MAX_PROCESSES // WINDOW_PER_HOUR) as exec:
        xs = exec.map(cache_worker, timestamps)
    
def retreive_satellite_data(tsList):
    """
    Attempts to pull the requested satellite data from the cache and if it
    does not exist, creates a cache file for it and then pulls it

    Parameters
    ----------
    tsList: list(datetime.datetime)
        Expects the tzinfo to be UTC and will retrieve a file with with exactly
        that timestamp
        
    Returns
    -------
    (np.ndarray, list(datetime.datetime))
    The first element in the tuple is a numpy array of shape (len(tsList), sat_lats, sat_lons, sat_bands=6)
    The second element is a list of the actual timestamps at which those images where taken (in UTC)

    Thus, arr[0] occurred at timestampList[0]
    
    """
    def retrieve_worker(ts):
        
        # Where to store the file
        yyyymmdd, hhmmss = parse_timestamp(ts)
        infile = f"{CACHE_DIR}/{yyyymmdd}/{hhmmss}.npz"
        
        # Try and get image from cache. If we can't, download it from aws first
        try:
            data = np.load(infile, allow_pickle = True)
        except Exception:
            cache_worker(ts)
            data = np.load(infile, allow_pickle = True)

        # timestamp gets wrapped in a numpy object on save so we call .item to unbox it
        return data["timestamp"].item(), data["image"]

    with ThreadPoolExecutor(max_workers=MAX_PROCESSES // WINDOW_PER_HOUR) as exec:
        stampsAndImages = exec.map(retrieve_worker, tsList)

    # Get images and stamps in their own arrays, merge the images
    [allStamps, allImages] = list(zip(*list(stampsAndImages)))
    return (np.vstack(allImages), list(allStamps))

def main():
    startTime = dt.datetime(2024, 2, 2, 0, 0, tzinfo=dt.UTC)
    endTime = dt.datetime(2024, 2, 3, 0, 0, tzinfo=dt.UTC)
    
    def generate_timestamps(
        start: dt.datetime = dt.datetime(2017, 3, 1, 0, 3, tzinfo=dt.UTC),
        end: dt.datetime = dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.UTC),
    ) -> list[dt.datetime]:
        """
        Generates a list of 5 minutes seperated datetimes starting on minute 3
        of each year 2018-2024 and 2017 without Jan and Feb

        Returns
        -------
        a list of datetimes in the range
        """

        timestamps = [[] for _ in range(12)]
        current_time = start

        while current_time < end:
            timestamps[current_time.minute // 5].append(current_time)
            current_time = current_time + dt.timedelta(minutes=5)

        return timestamps

    multiprocessing.set_start_method("forkserver", force=True)

    tsList = generate_timestamps(startTime, endTime)
    print(f"Generated timestamps {len(tsList)=}")
    
    start_t = time.time()
    with ThreadPoolExecutor(max_workers=WINDOW_PER_HOUR) as exec:
        exec.map(cache_images_from_aws, tsList)
    end_t = time.time()
    print(f"Completed from {startTime} to {endTime} in {end_t - start_t} seconds")

if __name__ =='__main__':
    main()