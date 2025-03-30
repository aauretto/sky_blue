# import pandas as pd
# import datetime as dt

# CSV_NAME = '/skyblue/PIREPcacheFullPickled.csv'

# df_big = pd.read_csv('/skyblue/test_cache.csv')
# df_small = pd.read_csv('/skyblue/fixesAndOopsPickleCache.csv')
# import sys
# print(sys.getsizeof(df_big))
# df_merged = pd.concat([df_big, df_small])
# df_sorted = df_merged.sort_values(by='Timestamp').reset_index(drop=True)
# df_sorted.to_csv(CSV_NAME, index=False)
