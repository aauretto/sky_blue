import keras
from goes2go import GOES
import satellite as st
import xarray as xr
import datetime as dt
from satellite_cache import retreive_satellite_data
import numpy as np
import matplotlib.pyplot as plt

SAT = GOES(satellite=16, product="ABI", domain="C")
BANDS =  [8, 9, 10, 13, 14, 15]
# SAVE_DIR = '/cluster/tufts/capstone25skyblue/models/2024_02_02-03_ex_machina_defaults/'
# MODEL_SAVE_PATH = SAVE_DIR + '2025_04_14_12_57model.keras'
SAVE_DIR = '/cluster/tufts/capstone25skyblue/models/2024_02_02-05_ex_machina_defaults/'
MODEL_SAVE_PATH = SAVE_DIR + 'simon_2024_02_02-052025_04_14_20_03model.keras'

def get_image(ts):
    timestamps = [ts - i * dt.timedelta(hours = 1) for i in range(9)]
    timestamps.reverse()
    try:
        xs, _ = retreive_satellite_data(timestamps)
    except:
        xs = [st.fetch(t, SAT) for t in timestamps]
        xs = xr.concat(xs, dim="t")
        lats, lons = st.calculate_coordinates(xs)
        xs = st.fetch_bands(xs, BANDS)
        xs = st.smooth(st.project(lats, lons, xs.data))
    print(xs.shape)
    return xs

ts = dt.datetime(2025, 2, 7, 0, 0, tzinfo=dt.UTC)
xs = get_image(ts)

inference_model = keras.models.load_model(MODEL_SAVE_PATH)
ys = inference_model.predict(np.array([xs]), batch_size=1)
frame = ys[0]
now_pred = frame[0]
alt_idx = 0
for alt_idx in range(14):
    alt_slice = now_pred[:, :, alt_idx]
    alt_mean = np.mean(alt_slice)
    alt_std = np.std(alt_slice)
    print(len(np.argwhere(alt_slice >= 0)))


    print(alt_slice.shape)
    plt.imshow(alt_slice, cmap='viridis', aspect='auto')  # or 'gray', 'hot', etc.
    plt.colorbar()  # adds a scale legend on the side
    plt.title(f"Altitude Level {alt_idx}")
    plt.xlabel("Longitudes")
    plt.ylabel("Latitudes")
    plt.savefig(f'/skyblue/alt_slice_{alt_idx}')
    plt.clf()