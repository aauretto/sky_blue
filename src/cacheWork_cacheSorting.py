"""
File: cacheSorting.py
Description:
    A quick way to merge two PIREP caches and keep the result sorted on timestamps

"""
import pandas as pd
import datetime as dt


CSV1 = 'path to the first CSV'
CSV2 = 'path to the second CSV'
CSV_SAVE = 'path to save the result csv to '

df_big = pd.read_csv(CSV1)
df_small = pd.read_csv(CSV2)

df_merged = pd.concat([df_big, df_small])
df_sorted = df_merged.sort_values(by='Timestamp').reset_index(drop=True)
df_sorted.to_csv(CSV_SAVE, index=False)
