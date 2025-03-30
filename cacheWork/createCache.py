# import pandas as pd
# import pirep as pr
# import datetime as dt
# from pirep.defs.aircraft import Aircraft
# from pirep.defs.spreading import spread_pirep, concatenate_all_pireps
# import numpy as np
# import sys
# import pickle
# import ast

# ### STEPS ###
# # 1) Read all pireps in date range
# # 2) Spread those pireps
# # 3) Append them to cache.csv


# CSV_FANME = "/skyblue/oopsCache.csv"
# ERR_LOG_FILE = '/skyblue/createCachePickleOopsErrorLog.txt'

# def create_cache(start, end):
#     print(f"Creating New Batch for {start} to {end}")
#     reports = pr.parse_all(pr.fetch(pr.url(start, end)))

#     # Need timestamps and spread grids

#     # Rip out timestamps for each report
#     timestamps = list(map(lambda r : r["Timestamp"], reports))


#     # Convert to pd
#     df = pd.DataFrame({
#         "Timestamp" : timestamps,
#         "Data"      : [pickle.dumps(r) for r in reports]     
#     })
    
#     df.to_csv(CSV_FANME, mode = "a", header=False, index=False)
#     # df.to_hdf(CSV_FANME, key="pireps", mode="a", complevel=9, complib="zlib", format="table")
# # parses all nones into actual stuff 
# def parse_negs():
#     ...

# def pull_from_cache(start, end):
#     # Reports has: (idx, val) OR (None, report)
#     reports = []

#     for chunk in pd.read_csv(CSV_FANME, chunksize=10):
#         # Look at last ts and if its before start continue
#         if pd.to_datetime(chunk['Timestamp'].iloc[len(chunk) - 1]).to_pydatetime() < start:
#             print("CONTINUING, last elem < Start")

#             continue
        
#         # Look at first ts and if its after end break
#         if pd.to_datetime(chunk['Timestamp'].iloc[0]).to_pydatetime() > end:
#             print("BREAKING, first elem > END")
#             break
        
#         # Read entire chunk if it is in range:
#         if pd.to_datetime(chunk['Timestamp'].iloc[0]).to_pydatetime() > start and \
#            pd.to_datetime(chunk['Timestamp'].iloc[len(chunk) - 1]).to_pydatetime() < end:
#             reports += chunk['Data'].tolist()
        
#         else: # find latest endpoint before end
#             start_idx = None
#             end_idx   = len(chunk)
#             for index, row in chunk.iterrows():
#                 # TODO: ask Tanay if we are incl or excl on the pirep time ranges
#                 if pd.to_datetime(row['Timestamp']).to_pydatetime() > start and start_idx is None:
#                     start_idx = index
#                 if pd.to_datetime(row['Timestamp']).to_pydatetime() > end:
#                     print("in end case: ", pd.to_datetime(row['Timestamp']).to_pydatetime())
#                     end_idx   = index
#                     break
#             print(f"slicing: [{start_idx}:{end_idx}]")
#             reports += chunk['Data'].iloc[start_idx:end_idx].tolist()
        
#     return reports
        


# if __name__ == "__main__":
#     if '--load' not in sys.argv:
#         # Intialize csv for appending later
#         if '--init' in sys.argv:
#             df = pd.DataFrame({
#                 "Timestamp" : [],
#                 "Data"      : []
#             })
#             df.to_csv(CSV_FANME, mode = "w", header=True, index=False)
#         else:
#             # final = dt.datetime(2025, 1, 1, 0, 20, 0, tzinfo=dt.UTC)
#             # start = dt.datetime(2017, 4, 5, 12,  0, 0, tzinfo=dt.UTC) 
#             # end   = dt.datetime(2017, 4, 6, 0, 0, 0, tzinfo=dt.UTC) 
#             final = dt.datetime(2017, 4, 5, 12) - dt.timedelta(milliseconds=1)
#             start = dt.datetime(2017, 2, 25, 0, 0)
#             end = start + dt.timedelta(hours=12)
            
#             with open(ERR_LOG_FILE, 'a') as errFile:
#                 while end < final:
#                     try:
#                         create_cache(start, end)
#                     except Exception as e:
#                         print(f"Issue creating cache on range: {start} - {end} with Error:\n {e}", file=errFile)
#                     diff = dt.timedelta(hours=12)
#                     start = end + dt.timedelta(milliseconds=1)
#                     end += diff

#                 try:
#                     create_cache(start, final)
#                 except Exception as e:
#                     print(f"Issue creating cache on range: {start} - {final} with Error:\n {e}", file=errFile)
                    
#             # final = dt.datetime(2025, 1, 1, 0, 20, 0, tzinfo=dt.UTC)
#             # start = dt.datetime(2017, 2, 25, 0,  0, 0, tzinfo=dt.UTC) 
#             # end   = dt.datetime(2017, 2,  25, 12, 0, 0, tzinfo=dt.UTC) 
#             # while end < final:
#             #     create_cache(start, end)
#             #     diff = dt.timedelta(hours=12)
#             #     start = end + dt.timedelta(milliseconds=1)
#             #     end += diff
#             # create_cache(start, final)

#     else:
#         df = pd.read_csv(CSV_FANME)
#         reps = [pickle.loads(ast.literal_eval(d)) for d in df['Data']]
#         print(type(reps))
#         print(reps[0])
#         grid = concatenate_all_pireps(reps[:15], 4e-5)
#         print(np.argwhere(~np.isnan(grid)))

    