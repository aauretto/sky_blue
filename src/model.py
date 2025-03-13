import datetime as dt
import multiprocessing

import keras
import keras_tuner as kt
import numpy as np
import tensorflow as tf
from goes2go import GOES
from numpy import typing as npt
from sklearn.model_selection import RandomizedSearchCV, train_test_split

import consts as consts
import pirep as pr
import satellite as st
from generator import Generator

BACKGROUND_RISKS = [0.01, 0.03, 0.05, 0.07]
BACKGROUND_RISK = BACKGROUND_RISKS[0]

def generate_combined_train_val_timestamps_set() -> list[dt.datetime]:
    """
    Generates a list of 5 minutes seperated datetimes starting on minute 3
    of each year 2018-2024 and 2017 without Jan and Feb

    Returns
    -------
    a list of datetimes in the range
    """
    years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    times = []
    for y in years:
        if y == 2025: # Heldout 2025 for testing
            break
        for m in months:
            if y == 2017 and (m == 1 or m == 2): # 2017 Jan and Feb don't have data
                continue
            current = dt.datetime(y, m, 1, 0, 3, 0)

            next_month = m + 1
            next_year = y
            if next_month > 12:
                next_month = 1
                next_year += 1
            next_month_start = dt.datetime(next_year, next_month, 1, 0, 0)
            while current < next_month_start:
                times.append(current)
                current = current + dt.timedelta(minutes=5)
    return times  

def generate_test_timestamps():
    """
    Generates a list of 5 minutes seperated datetimes starting on minute 3
    of 2025 ending on 2025-12-31

    Returns
    -------
    a list of datetimes in the range
    """
    years = [2025, 2026]
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    times = []
    for y in years:
        for m in months:
            current = dt.datetime(y, m, 1, 0, 3, 0)

            next_month = m + 1
            next_year = y
            if next_month > 12:
                next_month = 1
                next_year += 1
            next_month_start = dt.datetime(next_year, next_month, 1, 0, 0)
            recent_file_time = dt.datetime.now() - dt.timedelta(days=1) # 24 window for NOAA to upload the data to the AWS bucket
            if next_month_start > recent_file_time:
                next_month_start = recent_file_time
            while current < next_month_start:
                times.append(current)
                current = current + dt.timedelta(minutes=5)
    return times

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

    return dict(zip(timestamps, data)), timestamps


def get_labels(
    start: dt.datetime,
    end: dt.datetime,
) -> dict:
    # Retrieve PIREPs
    reports: list[dict] = pr.parse_all(pr.fetch(pr.url(start, end)))
    print("Parsed reports")
    # Convert reports to grids
    labels = dict(map(lambda row: (row["Timestamp"], row), reports))

    return labels


def model_initializer(hp):
    # Model parameters
    num_classes = 14
    out_steps = 4  # how many time units outward to predict
    lat_size = consts.GRID_RANGE["LAT"]
    lon_size = consts.GRID_RANGE["LON"]

    model = keras.Sequential()
    model.add(keras.layers.Input((out_steps, lat_size, lon_size, 6)))
    model.add(
        keras.layers.ConvLSTM2D(
            filters=hp.Int("filters", min_value=16, max_value=64, step=16),
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
        )
    )

    # Dropout for regularization
    model.add(
        keras.layers.Dropout(
            rate=hp.Float("dropout_rate", min_value=0.2, max_value=0.5, step=0.1)
        )
    )

    # 1x1 Conv2D to reduce feature map to num_classes
    model.add(
        keras.layers.Conv3D(
            filters=num_classes,
            kernel_size=(1, 1, 1),
            activation="linear",
            padding="same",
        )
    )

    # Compile the model
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float(
                "learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG"
            )
        ),
        metrics=[keras.metrics.MeanAbsoluteError()],
    )

    # model.summary()
    return model


def run_hyperparameter_tuning(
    train_dataset: keras.utils.Sequence,
    val_dataset: keras.utils.Sequence,
):
    tuner = kt.Hyperband(
        model_initializer,
        objective="val_loss",  # Minimize validation loss
        max_epochs=10,  # Maximum number of epochs per trial
        factor=3,  # The factor by which the number of trials decreases
        directory="kt_tuning",  # Directory to save results
        project_name="hyperparameter_tuning",
    )

    print("About to do the tuner search")
    tuner.search(train_dataset, epochs=10, validation_data=val_dataset, verbose=2)
    print("a best model has been retrieved")
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[
        0
    ].hyperparameters
    print(best_hyperparameters.values)
    best_model = tuner.get_best_models(num_models=1)
    print(best_model[0].summary())
    return best_model


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    timestamps = generate_combined_train_val_timestamps_set()
    # print(len(timestamps))
    # print("The timestamps for our sliced time period are", timestamps[808416:808416+12])
    # print(len(timestamps[808416:808416+12]))
    t_train, t_val = train_test_split(timestamps[808416:808416+12], test_size=0.20, random_state=42) # timestamp slice is for the desired subset to train with
    t_test = generate_test_timestamps()
    print(f"Length of train dataset: ", len(t_train))
    print(f"Length of val dataset: ", len(t_val))
    print(f"Length of test dataset: ", len(t_test))
    sat = GOES(satellite=16, product="ABI", domain="C")
    bands = [8, 9, 10, 13, 14, 15]

    train_generator = Generator(t_train, 2, 1, 1, BACKGROUND_RISK, 2, sat, bands) # xs shape: (4,1500,2500, 6) ys shape: (4,1500,2500, 14)
    val_generator = Generator(t_val, 2, 1, 1, BACKGROUND_RISK, 2, sat, bands)     # xs shape: (4,1500,2500, 6) ys shape: (4,1500,2500, 14)







    #******************************************************************#
    exit(0)
    #******************************************************************#




    start = dt.datetime(2024, 11, 6, 0, 0)
    end = dt.datetime(2024, 11, 6, 1, 0)

    data, timestamps = get_data(start, end)
    print("DATA has been retrieved")
    labels = get_labels(start, end)
    print("LABELS have been retrieved")

    t_train, t_test = train_test_split(timestamps, test_size=0.33, random_state=42)
    print(f"Number of training samples{len(t_train)}")  # 8
    print(f"Number of testing samples{len(t_test)}")  # 4

    # print("TRAIN-TEST-SPLIT created")
    t_train, t_val = train_test_split(t_train, test_size=0.2, random_state=42)
    print(f"Number of training samples{len(t_train)}")  # 6
    print(f"Number of validation samples{len(t_val)}")  # 2
    # print("TRAIN-VAL-SPLIT created")
    print("TRAINING GENERATOR")
    train_dataset = Generator(
        data, labels, t_train, 2, 1, 1, BACKGROUND_RISK, 2
    )  # xs shape: (4,1500,2500, 6) ys shape: (4,1500,2500, 14)
    print(f"Length of train dataset: {len(train_dataset)}")
    print("TESTING GENERATOR")
    test_dataset = Generator(data, labels, t_test, 2, 1, 1, BACKGROUND_RISK, 2)
    print(f"Length of test dataset: {len(test_dataset)}")
    print("VALIDATION GENERATIOB")
    val_dataset = Generator(data, labels, t_val, 2, 1, 1, BACKGROUND_RISK, 2)
    print(f"Length of val dataset: {len(val_dataset)}")

    print(
        f"train: {len(train_dataset)}, test: {len(test_dataset)}, val: {len(val_dataset)}"
    )
    # Code to test the model initializer
    # test_hp = kt.HyperParameters()
    # test_model = model_initializer(test_hp)
    # test_model.summary()

    # Code to test the data
    # sample_x, sample_y = train_dataset[0]
    # print("Sample X Shape", sample_x)
    # print("Sample Y Shape", sample_y)

    # Test a single training step
    # sample_x, sample_y = train_dataset[0]
    # model = model_initializer(kt.HyperParameters())
    # model.fit(sample_x, sample_y, epochs=1)
    #******************************************************************#

    print("About to run the hyperparameter tuning loop")
    best_model = run_hyperparameter_tuning(train_dataset, val_dataset)
    best_model.summary()

    checkpoint_path = "persistent_files/best_model_checkpoint.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",  # Monitor validation loss
        save_best_only=True,  # Save only the best model
        save_weights_only=False,  # Save entire model (structure + weights)
        verbose=1,
    )

    # best_model.fit(train_dataset, epochs=10, checkpoint_callback=[checkpoint_callback])
    # final_loss, final_mae = best_model.evaluate(X_test, y_test)
    # print(f"Test Loss: {final_loss}, Test MAE: {final_mae}")
