import datetime as dt

import keras
import numpy as np
import tensorflow as tf
from goes2go import GOES
from numpy import typing as npt
from sklearn.model_selection import train_test_split, RandomizedSearchCV

import consts as consts
import pirep as pr
import satellite as st

from generator import Generator
import keras_tuner as kt
from memory_profiler import profile

BACKGROUND_RISKS = [0.01, 0.03, 0.05, 0.07]
BACKGROUND_RISK = BACKGROUND_RISKS[0]


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
    model.add(
        keras.layers.ConvLSTM2D(
            filters=hp.Int("filters", min_value=16, max_value=64, step=16),
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            input_shape=(out_steps, lat_size, lon_size, 6),
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

@profile
def run_hyperparameter_tuning(
    train_dataset: keras.utils.PyDataset,
    val_dataset: keras.utils.PyDataset,
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
    tuner.search(train_dataset, epochs=1, validation_data=val_dataset)
    print("a best model has been retrieved")
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters

    print("Best Hyperparameters:", best_hyperparameters.values)
    return best_model


def run_hyperparameter_tuning_skelarn(
    train_dataset: keras.utils.PyDataset,
    val_dataset: keras.utils.PyDataset,
):
    print("Splitting data for train")
    X_train, y_train = train_dataset
    print("Splitting data for test")
    X_val, y_val = val_dataset
    "Initializing"
    model = model_initializer()
    print("About to do the tuner search")
    param_distributions = {
        'batch_size': [16, 32, 64],
        'epochs': [5, 10, 20],
        'optimizer': ['adam', 'rmsprop'],
        'learning_rate': [0.001, 0.01, 0.1],
    }
    
    # Run random search
    tuner = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=10,  # Number of different settings to try
        scoring='neg_mean_squared_error',  # Optimization metric
        cv=3,  # Cross-validation folds
        verbose=1,
        n_jobs=-1
    )
    print("Starting hyperparameter search...")
    tuner.fit(X_train, y_train, validation_data=(X_val, y_val))
    
    print("Best model retrieved")
    best_model = tuner.best_estimator_
    best_hyperparameters = tuner.best_params_
    
    print("Best Hyperparameters:", best_hyperparameters)
    return best_model

if __name__ == "__main__":
    start = dt.datetime(2024, 11, 6, 0, 0)
    end = dt.datetime(2024, 11, 6, 1, 0)

    data, timestamps = get_data(start, end)
    print("DATA has been retrieved")
    labels = get_labels(start, end)
    print("LABELS have been retrieved")

    t_train, t_test = train_test_split(timestamps, test_size=0.33, random_state=42)
    print(f"Number of training samples{len(t_train)}") # 8 
    print(f"Number of testing samples{len(t_test)}") # 4

    # print("TRAIN-TEST-SPLIT created")
    t_train, t_val = train_test_split(t_train, test_size=0.2, random_state=42)
    print(f"Number of training samples{len(t_train)}") #  6
    print(f"Number of validation samples{len(t_val)}") # 2
    # print("TRAIN-VAL-SPLIT created")
    print("TRAINING GENERATOR")
    train_dataset = Generator(
        data, labels, t_train, 4, 1, 2, BACKGROUND_RISK
    )  # xs shape: (4,1500,2500, 6) ys shape: (4,1500,2500, 14)
    print(f"Length of train dataset: {len(train_dataset)}")
    print("TESTING GENERATOR")
    test_dataset = Generator(data, labels, t_test, 4, 1, 2, BACKGROUND_RISK)
    print(f"Length of test dataset: {len(test_dataset)}")
    print("VALIDATION GENERATIOB")
    val_dataset = Generator(data, labels, t_val, 4, 1, 2, BACKGROUND_RISK)
    print(f"Length of val dataset: {len(val_dataset)}")

    print(
        f"train: {len(train_dataset)}, test: {len(test_dataset)}, val: {len(val_dataset)}"
    )
    print("About to run the hyperparameter tuning loop")
    best_model = run_hyperparameter_tuning(train_dataset, val_dataset)
    best_model.summary()
    best_model.fit(train_dataset, epochs=10, batch_size=32)
    # final_loss, final_mae = best_model.evaluate(X_test, y_test)
    # print(f"Test Loss: {final_loss}, Test MAE: {final_mae}")
