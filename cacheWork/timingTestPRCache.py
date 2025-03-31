import pandas as pd
import pirep as pr
import datetime as dt
import numpy as np

import time

start = pd.to_datetime(dt.datetime(2023, 12, 1, 0, 0, tzinfo=dt.UTC))
end = pd.to_datetime(dt.datetime(2023, 12, 1, 0, 30, tzinfo=dt.UTC))
df = pd.read_csv('/skyblue/PIREPcacheFull.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)

cacheStart = time.time()
start_idx = np.searchsorted(df['Timestamp'], start, side='left')  # Find the index where `start` should go
end_idx = np.searchsorted(df['Timestamp'], end, side='right')
res = df.iloc[start_idx:end_idx]
cacheEnd = time.time()
pullStart = time.time()
resPull = pr.parse_all(pr.fetch(pr.url(start, end)))
pullEnd = time.time()

print(f"Cache searching took {cacheEnd - cacheStart} and found {len(res)} PIREPS")
print(f"Pull searching took {pullEnd - pullStart} and found {len(resPull)} PIREPS")
