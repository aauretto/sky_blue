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
import psutil
import os

<<<<<<< Updated upstream
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss  # Resident Set Size (RSS) in bytes

BACKGROUND_RISKS = [0.01, 0.03, 0.05, 0.07]
BACKGROUND_RISK = BACKGROUND_RISKS[0]
=======
def get_memory_info():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        "RSS (Resident Set Size)": f"{mem_info.rss / 1024 ** 2:.2f} MB",
        "VMS (Virtual Memory Size)": f"{mem_info.vms / 1024 ** 2:.2f} MB",
        "Shared Memory": f"{mem_info.shared / 1024 ** 2:.2f} MB",
        "Text (code)": f"{mem_info.text / 1024 ** 2:.2f} MB",
        "Data + Stack": f"{mem_info.data / 1024 ** 2:.2f} MB",
        "Dirty Pages": f"{mem_info.dirty / 1024 ** 2:.2f} MB",
        "PID": f"{os.getpid()}"
    }
>>>>>>> Stashed changes


def get_data(
    start: dt.datetime,
    end: dt.datetime,
    sat: GOES = GOES(satellite=16, product="ABI", domain="C"),
    bands: list[int] = [8, 9, 10, 13, 14, 15],
) -> tuple[npt.NDArray, npt.ArrayLike]:
    # Fetch satellite data
    print(f"\n\n\n************************* get_data start Mem = {get_memory_usage()}*************************\n\n")
    data = st.fetch_range(start, end, satellite=sat)
    print(f"\n\n\n************************* get_data fetched range Mem = {get_memory_usage()}*************************\n\n")

    # Project data onto grid
    lats, lons = st.calculate_coordinates(data)
    print(f"\n\n\n************************* get_data calculate_coordinates range Mem = {get_memory_usage()}*************************\n\n")
    
    band_data = st.fetch_bands(data, bands)
    print(f"\n\n\n************************* get_data fetch bands range Mem = {get_memory_usage()}*************************\n\n")
    timestamps = np.array(band_data.coords["t"], dtype=np.datetime64)
    print(f"\n\n\n************************* get_data timestamps np array allocated range Mem = {get_memory_usage()}*************************\n\n")
    data = st.smooth(st.project(lats, lons, band_data.data))
    print(f"\n\n\n************************* get_data data has been smoothened and projected Mem = {get_memory_usage()}*************************\n\n")

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
    print(f"\n\n\n************************* get_labels start Mem = {get_memory_usage()}*************************\n\n")
    reports: list[dict] = pr.parse_all(pr.fetch(pr.url(start, end)))
    print("Parsed reports")
    # Convert reports to grids
    labels = dict(map(lambda row: (row["Timestamp"], row), reports))
    print(f"\n\n\n************************* get_labels end Mem = {get_memory_usage()}*************************\n\n")

    return labels


def model_initializer(hp):
    print(f"\n\n\n************************* model_initializer start Mem = {get_memory_usage()}*************************\n\n")
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
    print(f"\n\n\n************************* model_initializer end Mem = {get_memory_usage()}*************************\n\n")
    return model


def run_hyperparameter_tuning(
    train_dataset: keras.utils.Sequence,
    val_dataset: keras.utils.Sequence,
):
    print(f"\n\n\n************************* Start run_hyperparameter_tuning Mem = {get_memory_usage()}*************************\n\n")
    tuner = kt.Hyperband(
        model_initializer,
        objective="val_loss",  # Minimize validation loss
        max_epochs=10,  # Maximum number of epochs per trial
        factor=3,  # The factor by which the number of trials decreases
        directory="kt_tuning",  # Directory to save results
        project_name="hyperparameter_tuning",
    )
    print(f"\n\n\n*************************  run_hyperparameter_tuning KT Tuner initialized Mem = {get_memory_usage()}*************************\n\n")

    print(f"About to do the tuner search{get_memory_info()}")
    tuner.search(train_dataset, epochs=10, validation_data=val_dataset, verbose=2)
<<<<<<< Updated upstream
    print(f"\n\n\n*************************  KT Tuner Search has run Mem = {get_memory_usage()}*************************\n\n")
    print("a best model has been retrieved")
=======
    print(f"a best model has been retrieved{get_memory_info()}")
>>>>>>> Stashed changes
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[
        0
    ].hyperparameters
    print(f"\n\n\n*************************   run_hyperparameter_tuning best hyperparems retrieved Mem = {get_memory_usage()}*************************\n\n")
    print(best_hyperparameters.values)
    best_model = tuner.get_best_models(num_models=1)
    print(f"\n\n\n*************************   run_hyperparameter_tuning best model retrieved Mem = {get_memory_usage()}*************************\n\n")
    print(best_model[0].summary())
    return best_model


if __name__ == "__main__":
<<<<<<< Updated upstream
    print(f"\n\n\n************************* Start program Mem = {get_memory_usage()}*************************\n\n")
    multiprocessing.set_start_method("forkserver", force=False)

    start = dt.datetime(2024, 11, 6, 0, 0)
    end = dt.datetime(2024, 11, 6, 1, 0)
    print(f"\n\n\n************************* Datetime made Mem = {get_memory_usage()}*************************\n\n")
    data, timestamps = get_data(start, end)
    print(f"\n\n\n************************* Data retrieved Mem = {get_memory_usage()}*************************\n\n")
    print("DATA has been retrieved")
    labels = get_labels(start, end)
    print(f"\n\n\n************************* Labels retrieved Mem = {get_memory_usage()}*************************\n\n")
    print("LABELS have been retrieved")

    t_train, t_test = train_test_split(timestamps, test_size=0.33, random_state=42)
    print(f"\n\n\n************************* Train test for train and test split Mem = {get_memory_usage()}*************************\n\n")
    print(f"Number of training samples{len(t_train)}")  # 8
    print(f"Number of testing samples{len(t_test)}")  # 4

    # print("TRAIN-TEST-SPLIT created")
    t_train, t_val = train_test_split(t_train, test_size=0.2, random_state=42) # t_train: timestamps
    print(f"\n\n\n************************* Train test split for train and val Mem = {get_memory_usage()}*************************\n\n")
    print(f"Number of training samples{len(t_train)}")  # 6
    print(f"Number of validation samples{len(t_val)}")  # 2
    # print("TRAIN-VAL-SPLIT created")
    print("TRAINING GENERATOR")
    train_dataset = Generator(
        data, labels, t_train, 2, 1, 1, BACKGROUND_RISK, 2
    )  # xs shape: (4,1500,2500, 6) ys shape: (4,1500,2500, 14)
    print(f"\n\n\n************************* Train generator memory check = {get_memory_usage()}*************************\n\n")
    print(f"Length of train dataset: {len(train_dataset)}")
    print("TESTING GENERATOR")
    test_dataset = Generator(data, labels, t_test, 2, 1, 1, BACKGROUND_RISK, 2)
    print(f"\n\n\n************************* Test generator memory check = {get_memory_usage()}*************************\n\n")
    print(f"Length of test dataset: {len(test_dataset)}")
    print("VALIDATION GENERATIOB")
    val_dataset = Generator(data, labels, t_val, 2, 1, 1, BACKGROUND_RISK, 2)
    print(f"\n\n\n************************* Val generator memory check = {get_memory_usage()}*************************\n\n")
    print(f"Length of val dataset: {len(val_dataset)}")

    print(
        f"train: {len(train_dataset)}, test: {len(test_dataset)}, val: {len(val_dataset)}"
=======
    multiprocessing.set_start_method("forkserver", force=True)
    np.random.seed(42)
    timestamps = generate_timestamps(
        start=dt.datetime(2024, 11, 6, 0, 0, tzinfo=dt.UTC),
        end=dt.datetime(2024, 11, 6, 12, 0, tzinfo=dt.UTC),
>>>>>>> Stashed changes
    )
    # Code to test the model initializer
    # test_hp = kt.HyperParameters()
    # test_model = model_initializer(test_hp)
    # test_model.summary()
<<<<<<< Updated upstream

    # Code to test the data
    # sample_x, sample_y = train_dataset[0]
    # print("Sample X Shape", sample_x)
    # print("Sample Y Shape", sample_y)

    # Test a single training step
    # sample_x, sample_y = train_dataset[0]
    # model = model_initializer(kt.HyperParameters())
    # model.fit(sample_x, sample_y, epochs=1)

    print("About to run the hyperparameter tuning loop")
    print(f"\n\n\n************************* Right before hyperparameter tuning memory check = {get_memory_usage()}*************************\n\n")
    best_model = run_hyperparameter_tuning(train_dataset, val_dataset) # TODO checkpoint the best model returned by KT Tuner
    print(f"\n\n\n************************* Right after hyperparameter tuning memory check = {get_memory_usage()}*************************\n\n")
=======
    print(f"About to run the hyperparameter tuning loop \n{get_memory_info()}")
    best_model = run_hyperparameter_tuning(train_generator, val_generator)
>>>>>>> Stashed changes
    best_model.summary()

    print(f"\n\n\n************************* Checkpoint about to be created memory check = {get_memory_usage()}*************************\n\n")
    checkpoint_path = "persistent_files/best_model_checkpoint.h5"
    print(f"\n\n\n************************* Checkpoint path created memory check = {get_memory_usage()}*************************\n\n")
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",  # Monitor validation loss
        save_best_only=True,  # Save only the best model
        save_weights_only=False,  # Save entire model (structure + weights)
        verbose=1,
    )
<<<<<<< Updated upstream
    print(f"\n\n\n************************* Checkpoint callback created memory check = {get_memory_usage()}*************************\n\n")

    best_model.fit(train_dataset, epochs=10, checkpoint_callback=[checkpoint_callback])
    print(f"\n\n\n************************* Best model has been fit we are done created memory check = {get_memory_usage()}*************************\n\n")
=======
    print("About to fit the best model to the fuller dataset and about to enter TODO section")
    best_model.fit(
        train_generator, epochs=10, checkpoint_callback=[checkpoint_callback]
    ) # TODO maybe replace with fit_generator
>>>>>>> Stashed changes
    # final_loss, final_mae = best_model.evaluate(X_test, y_test)
    # print(f"Test Loss: {final_loss}, Test MAE: {final_mae}")
