1) Convert all numpy arrays that are float64s to float32s
2) Correctly frame timestamp data
    Sort timestamps based on the minute
    Hold out 2 rows as validation
    Other 10 rows are train
3) Generator takes a list of timestamps as its data
    - already implemented in the main of model.py
4) to compute a batch in generator
    - get batch_size timestamps
    - for each timestamp generate the frame
        - add 1, 2,3, .. 8 hours to create a list of 9 timestamps
    - shape of getitem output is (batch_size, out_times = 9, datashape (x=(1500, 2500, 6) y=(1500, 2500, 14)))
5) Remove background risk as it is now randomized