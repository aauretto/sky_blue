import datetime as dt
import multiprocessing

import keras
import keras_tuner as kt
import numpy as np
from numpy import random as rnd
from goes2go import GOES
from numpy import typing as npt

import consts as consts
import pirep as pr
import satellite as st
from generator import Generator


def generate_timestamps(
    start: dt.datetime = dt.datetime(2017, 3, 1, 0, 3, tzinfo=dt.UTC),
    end: dt.datetime = dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.UTC),
) -> list[dt.datetime]:
    """
    Generates a list of 5 minutes seperated datetimes starting on minute 3
    of each year 2018-2024 and 2017 without Jan and Feb

    Returns
    -------
    a list of datetimes in the range
    """

    timestamps = [[] for _ in range(12)]
    current_time = start

    while current_time < end:
        timestamps[current_time.minute // 5].append(current_time)
        current_time = current_time + dt.timedelta(minutes=5)

    return timestamps


def model_initializer(hp, frame_size: int = 9):
    # Model parameters
    num_classes = 14
    lat_size = consts.GRID_RANGE["LAT"]
    lon_size = consts.GRID_RANGE["LON"]

    model = keras.Sequential()
    model.add(keras.layers.Input((frame_size, lat_size, lon_size, 6))) # 6 is number of bands 
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
    timestamps = generate_timestamps(
        start=dt.datetime(2024, 11, 6, 0, 0, tzinfo=dt.UTC),
        end=dt.datetime(2024, 11, 6, 12, 0, tzinfo=dt.UTC),
    )
    # print([len(sequence) for sequence in timestamps])
    # print([sequence[0] for sequence in timestamps])
    # print("\n\n\n\n")
    # print([sequence[90] for sequence in timestamps])

    rng = rnd.default_rng(seed=42)
    rng.shuffle(timestamps)
    t_train, t_val = timestamps[:10], timestamps[10:]
    t_test = generate_timestamps(
        start=dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.UTC),
        end=dt.datetime.now(tz=dt.UTC),
    )

    print(f"Length of train dataset: {len(t_train)} with {sum(map(lambda x: len(x), t_train))} total elements")
    print(f"Length of val dataset: {len(t_val)} with {sum(map(lambda x: len(x), t_val))} total elements")
    print(f"Length of test dataset: {len(t_test)} with {sum(map(lambda x: len(x), t_test))} total elements")

    sat = GOES(satellite=16, product="ABI", domain="C")
    bands = [8, 9, 10, 13, 14, 15]

    train_generator = Generator(
        t_train, batch_size=1, frame_size=9, sat=sat, bands=bands
    )  # xs shape: (4,1500,2500, 6) ys shape: (4,1500,2500, 14)
    val_generator = Generator(
        t_val, batch_size=1, frame_size=9, sat=sat, bands=bands
    )  # xs shape: (4,1500,2500, 6) ys shape: (4,1500,2500, 14)
    test_generator = Generator(
        t_test, batch_size=1, frame_size=9, sat=sat, bands=bands
    )  # xs shape: (4,1500,2500, 6) ys shape: (4,1500,2500, 14)

    print("train batches: ", len(train_generator))
    print("val batches: ", len(val_generator))
    print("test batches: ", len(test_generator))

    # Code to test the model initializer
    # test_hp = kt.HyperParameters()
    # test_model = model_initializer(test_hp, frame_size=9)
    # test_model.summary()

    print("About to run the hyperparameter tuning loop")
    best_model = run_hyperparameter_tuning(train_generator, val_generator)
    best_model.summary()

    checkpoint_path = "persistent_files/best_model_checkpoint.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",  # Monitor validation loss
        save_best_only=True,  # Save only the best model
        save_weights_only=False,  # Save entire model (structure + weights)
        verbose=1,
    )

    best_model.fit(
        train_generator, epochs=10, checkpoint_callback=[checkpoint_callback]
    )
    # final_loss, final_mae = best_model.evaluate(X_test, y_test)
    # print(f"Test Loss: {final_loss}, Test MAE: {final_mae}")
