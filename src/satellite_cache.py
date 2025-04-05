from goes2go import GOES
import datetime as dt
import satellite as st
import multiprocessing
import numpy as np 
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from model import generate_timestamps #TODO this func should be in satellite

import os
import xarray as xr

### MEGA CONSTANTS:
MAX_THREADS = 1280
WINDOW_PER_HOUR = 12
BANDS = [8,9,10,13,14,15]
CACHE_DIR = "/skyblue/caching"


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

SAT = GOES(satellite=16, product="ABI", domain="C")


# Worker function that will download a single GOES-16 image and save it with an updated timstamp
# in the cache
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
    # Figure out where we are going to store the cached file
    yyyymmdd, hhmmss = parse_timestamp(ts)
    thisDir = f"{CACHE_DIR}/{yyyymmdd}"
    outfile = f"{thisDir}/{hhmmss}.npz"

    if os.path.isfile(outfile):
        return

    # Get data from AWS, compute updated timestamp for data, pull out bands we want
    rawXarr = st.fetch(ts, SAT)
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
    
    # Write file to disk
    os.makedirs(thisDir, exist_ok=True)
    np.savez(outfile, image=smoothed, timestamp=updated_timestamp)
 

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
    with ThreadPoolExecutor(max_workers=MAX_THREADS // WINDOW_PER_HOUR) as exec:
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
        except FileNotFoundError:
            cache_worker(ts)
            data = np.load(infile, allow_pickle = True)

        # timestamp gets wrapped in a numpy object on save so we call .item to unbox it
        return data["timestamp"].item(), data["image"]

    with ThreadPoolExecutor(max_workers=MAX_THREADS // WINDOW_PER_HOUR) as exec:
        stampsAndImages = exec.map(retrieve_worker, tsList)

    # Get images and stamps in their own arrays, merge the images
    [allStamps, allImages] = list(zip(*list(stampsAndImages)))
    return (np.vstack(allImages), list(allStamps))


if __name__ =='__main__':
    multiprocessing.set_start_method("forkserver", force=True)

    startTime = dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.UTC)
    endTime = dt.datetime(2024, 1, 11, 0, 0, tzinfo=dt.UTC)

    tsList = generate_timestamps(startTime, endTime) #TODO make in st
    
    start_t = time.time()
    with ThreadPoolExecutor(max_workers=WINDOW_PER_HOUR) as exec:
        exec.map(cache_images_from_aws, tsList)
    end_t = time.time()
    print(f"Completed from {startTime} to {endTime} in {end_t - start_t} seconds")
