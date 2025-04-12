import ast
import pandas as pd
import datetime as dt
import pickle
import numpy as np
import time
import sys
from pirep.defs.spreading import concatenate_all_pireps
import psutil
import gc

def read_pirep_cache() -> tuple[list[pd._libs.tslibs.timestamps.Timestamp], list[dict]]:
    _CSV_NAME = '/cluster/tufts/capstone25skyblue/Caches/PIREPcacheFullPickled.csv'
    df = pd.read_csv(_CSV_NAME)
    times = pd.to_datetime(df['Timestamp'], utc=True)
    reports = [pickle.loads(ast.literal_eval(d)) for d in df['Data']]
    del df
    gc.collect()
    return times, reports

def retrieve_from_pirep_cache(start: dt.datetime, end: dt.datetime, 
                              times: list[pd._libs.tslibs.timestamps.Timestamp], reports: list[dict] = None) -> list[dict]:
    start_idx = np.searchsorted(times, start, side='left')
    end_idx = np.searchsorted(times, end, side='right')
    return start_idx, end_idx



if __name__ == "__main__":
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