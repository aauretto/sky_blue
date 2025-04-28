import ast
import pandas as pd
import datetime as dt
import pickle
import numpy as np
import time
import sys
from pirep import concatenate_all_pireps
import psutil
import gc

def read_pirep_cache(csvName = '/cluster/tufts/capstone25skyblue/Caches/PIREPcacheFullPickled.csv') -> tuple[list[pd._libs.tslibs.timestamps.Timestamp], list[dict]]:
    """
    Pulls in the PIREP cache from its location on disk and returns the relevant data as 
    versatile lists

    Parameters
    ----------
    csvName: str
        The path to the csv to read

    Returns
    -------
    times: list[pd._libs.tslibs.timestamps.Timestamp]
        The timestamps from the cache
    reports: list[dict]
        the pireps from the cache
    """
    df = pd.read_csv(csvName)
    times = pd.to_datetime(df['Timestamp'], utc=True)
    reports = [pickle.loads(ast.literal_eval(d)) for d in df['Data']]
    del df
    gc.collect()
    return times, reports

def retrieve_from_pirep_cache(start: dt.datetime, end: dt.datetime, 
                              times: list[pd._libs.tslibs.timestamps.Timestamp], reports: list[dict] = None) -> list[dict]:
    """
    Returns the start and end indices of the place in times which correspond to [start, end)

    Parameters
    ----------
    start: dt.datetime
        The start time of the PIREPs to look for
    end: dt.datetime
        The end time of the PIREPs to look for
    times: list[pd._libs.tslibs.timestamps.Timestamp]
        The times to search through
    reports: list[dict]
        An unused parameter that could in theory be used to return the actually relevant pireps

    """
    start_idx = np.searchsorted(times, start, side='left')
    end_idx = np.searchsorted(times, end, side='right')
    return start_idx, end_idx



if __name__ == "__main__":

    """
    Some profiling we did of the cache we made
    """

    def get_memory_usage():
        """Returns the current memory usage of the process in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

    CSV_FNAME = "/skyblue/PIREPcacheFullPickled.csv"

    print(f"About to read in CSV")
    start_t = time.time()
    df = pd.read_csv(CSV_FNAME)
    end_t = time.time()
    print(f"Finished reading in CSV in {end_t - start_t} time and of size {sys.getsizeof(df)} with the program using {get_memory_usage()} MB")

    print("About to convert times to list")
    start_t = time.time()
    times = pd.to_datetime(df['Timestamp'], utc=True)
    end_t = time.time()
    print(f"Finished converting times in {end_t - start_t} time with size of {sys.getsizeof(times)} with the program using {get_memory_usage()} MB")

    print("About to convert pireps")
    start_t = time.time()
    reports = [pickle.loads(ast.literal_eval(d)) for d in df['Data']]
    end_t = time.time()
    print(f"Finished converting reports in {end_t - start_t} time with size of {sys.getsizeof(reports)} with the program using {get_memory_usage()} MB")

    print("About to concat 10 pireps")
    start_t = time.time()
    grid = concatenate_all_pireps(reports[:10], 4e-5)
    end_t = time.time()
    print(f"Finished concat-ing all pireps in {end_t - start_t} time with size {sys.getsizeof(grid)} with the program using {get_memory_usage()} MB")


    del df
    gc.collect()
    print(f"After delling df the program is using {get_memory_usage()} MB")