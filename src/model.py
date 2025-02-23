from goes2go import GOES

from sklearn.model_selection import train_test_split
import datetime as dt
import numpy as np
from numpy import typing as npt
import pandas as pd
import keras
import pirep as pr
import satellite as st
import consts as consts
import tensorflow as tf


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


def get_labels(
    start: dt.datetime,
    end: dt.datetime,
    num_frames: int,
    timestamps: npt.ArrayLike,
) -> npt.NDArray:
    # Retrieve PIREPs
    reports: pd.DataFrame = pr.parse_all(pr.fetch(pr.url(start, end)))

    # Convert reports to grids
    grids = pd.DataFrame(
        {
            "Timestamp": reports["Timestamp"].to_numpy(),
            "Grid": reports.apply(pr.compute_grid, axis=1).apply(lambda x: x[0]),
        }
    )

    labels = np.zeros(
        (
            num_frames,
            consts.GRID_RANGE["LAT"],
            consts.GRID_RANGE["LON"],
            consts.GRID_RANGE["ALT"],
        )
    )
    for i, timestamp in enumerate(timestamps[:1]):
        mask = np.abs((grids["Timestamp"] - timestamp) / pd.Timedelta(minutes=1)) <= 15
        window = np.array(grids.loc[mask]["Grid"].tolist())
        binned_window = np.max(window, axis=0)  # TODO: Spread the PIREPs here
        labels[i] = binned_window

    assert labels.shape == (
        num_frames,
        consts.GRID_RANGE["LAT"],
        consts.GRID_RANGE["LON"],
        consts.GRID_RANGE["ALT"],  # TODO: Change when the altitude range is modified
    )

    return labels


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
    labels = get_labels(start, end, data.shape[0], timestamps)

    data_windows = get_windows(data, 4, 2)
    label_windows = get_windows(labels, 4, 2)

    # MODEL TRAINING
    X = data_windows
    y = label_windows
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    model = model_initializer()
    print(f"X: {X.shape}, y: {y.shape}")
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # model.evaluate(): To calculate the loss values for the input data
    # model.predict(): To generate network output for the input data
