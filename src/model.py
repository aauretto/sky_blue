import datetime as dt

import keras
import numpy as np
import tensorflow as tf
from goes2go import GOES
from numpy import typing as npt
from sklearn.model_selection import train_test_split

import consts as consts
import pirep as pr
import satellite as st
from pirep.defs.spreading import concatenate_all_pireps

BACKGROUND_RISK = 0.01


def get_data(
    start: dt.datetime,
    end: dt.datetime,
    sat: GOES = GOES(satellite=16, product="ABI", domain="C"),
    bands: list[int] = [8, 9, 10, 13, 14, 15],
) -> tuple[npt.NDArray, npt.ArrayLike]:
    # Fetch satellite data
    data = st.fetch_range(start, end, satellite=sat)

    # Project data onto grid
    lats, lons = st.calculate_coordinates(data)
    band_data = st.fetch_bands(data, bands)
    timestamps = np.array(band_data.coords["t"], dtype=np.datetime64)
    data = st.smooth(st.project(lats, lons, band_data.data))

    assert data.shape[1:] == (
        consts.GRID_RANGE["LAT"],
        consts.GRID_RANGE["LON"],
        len(bands),
    )

    return data, timestamps


def bin_labels(frames: dict, num_frames: int, timestamps: npt.ArrayLike) -> npt.NDArray:
    labels = np.zeros(
        (
            num_frames,
            consts.GRID_RANGE["LAT"],
            consts.GRID_RANGE["LON"],
            consts.GRID_RANGE["ALT"],
        )
    )

    for i, timestamp in enumerate(timestamps[:1]):
        filtered_frames = {
            frame_timestamp: frame
            for frame_timestamp, frame in frames.items()
            if (frame_timestamp - timestamp) / dt.timedelta(minutes=1)
            <= consts.PIREP_RELEVANCE_DURATION
        }

        window = list(filtered_frames.values())
        binned_window = concatenate_all_pireps(window, BACKGROUND_RISK)
        labels[i] = binned_window

    return labels


# TODO make this a full and complete wrapper
def get_labels(
    start: dt.datetime,
    end: dt.datetime,
    num_frames: int,
    timestamps: npt.ArrayLike,
) -> npt.NDArray:
    # Retrieve PIREPs
    reports: list[dict] = pr.parse_all(pr.fetch(pr.url(start, end)))
    print("Parsed reports")
    # Convert reports to grids
    frames = dict(map(lambda row: (row["Timestamp"], row), reports))
    labels = bin_labels(frames, num_frames, timestamps)
    assert labels.shape == (
        num_frames,
        consts.GRID_RANGE["LAT"],
        consts.GRID_RANGE["LON"],
        consts.GRID_RANGE["ALT"],
    )

    return labels.reshape(num_frames, -1)


def get_windows(data: npt.NDArray, width: int, offset: int):
    # data.shape = (n, 1500, 2500, 6)
    # data_output.shape = (k, (w, 1500, 2500, 6))
    num_frames = data.shape[0]
    num_windows = (num_frames - width) // offset + 1
    data_windows = np.array(
        [data[idx * offset : idx * offset + width, :] for idx in range(num_windows)]
    )

    return data_windows


def model_initializer():
    # Model parameters
    num_classes = 14
    out_steps = 4  # how many time units outward to predict

    model = keras.Sequential(
        [
            # Shape => [batch, out_steps, lats * lons * alts].
            keras.layers.Reshape(
                [
                    out_steps,
                    consts.GRID_RANGE["LAT"] * consts.GRID_RANGE["LON"] * 6,
                ]
            ),
            # Shape [batch, time, features] => [batch, lstm_units].
            # Adding more `lstm_units` just overfits more quickly.
            keras.layers.LSTM(
                units=consts.GRID_RANGE["LAT"] * consts.GRID_RANGE["LON"] * num_classes
            ),
            # Shape => [batch, out_steps*features].
            keras.layers.Dense(
                consts.GRID_RANGE["LAT"] * consts.GRID_RANGE["LON"] * num_classes,
                kernel_initializer=tf.zeros_initializer(),
            ),
            # Shape => [batch, out_steps, features].
            keras.layers.Reshape(
                [
                    out_steps,
                    consts.GRID_RANGE["LAT"],
                    consts.GRID_RANGE["LON"],
                    num_classes,
                ]
            ),
        ]
    )

    # Compile the model
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.MeanAbsoluteError()],
    )

    # model.summary()
    return model


if __name__ == "__main__":
    start = dt.datetime(2024, 11, 6, 0, 0)
    end = dt.datetime(2024, 11, 6, 1, 0)

    data, timestamps = get_data(start, end)
    print("DATA has been retrieved")
    labels = get_labels(start, end, data.shape[0], timestamps)
    print("LABELS have been retrieved")
    data_windows = get_windows(data, 4, 2)
    label_windows = get_windows(labels, 4, 2)
    print("WINDOWS have been retrieved")
    # MODEL TRAINING
    X = data_windows
    y = label_windows
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    print("TRAIN-TEST-SPLIT created")

    model = model_initializer()
    print(f"X: {X.shape}, y: {y.shape}")
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # model.evaluate(): To calculate the loss values for the input data
    # model.predict(): To generate network output for the input data
