from Logger import LOGGER
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
import random


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


def model_initializer(num_filters, dropout_rate, learning_rate, frame_size: int = 9):
    # Model parameters
    num_classes = 14
    lat_size = consts.GRID_RANGE["LAT"]
    lon_size = consts.GRID_RANGE["LON"]

    model = keras.Sequential()
    model.add(keras.layers.Input((frame_size, lat_size, lon_size, 6)))
    model.add(
        keras.layers.ConvLSTM2D(
            filters=num_filters,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
        )
    )

    # Dropout for regularization
    model.add(
        keras.layers.Dropout(
            rate=dropout_rate
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
            learning_rate=learning_rate
        ),
        metrics=[keras.metrics.MeanAbsoluteError()],
    )

    # model.summary()
    return model

def sample_hyperparameters():
    filters = random.choice([16, 32, 64])  # Example filter sizes
    dropout_rate = random.uniform(0.2, 0.5)
    learning_rate = random.choice([1e-4, 1e-3, 1e-2])
    return filters, dropout_rate, learning_rate

def run_hyperparameter_tuning(train_generator, val_generator):
    max_epochs = 50
    max_trials = 16
    batch_size = 1
    results = []

    epochs_per_trial = 5
    num_trials = max_trials
    while num_trials > 0:
        print(f"\nRunning stage with {num_trials} trials for {epochs_per_trial} epochs...")
        for _ in range(num_trials):
            filters, dropout_rate, learning_rate = sample_hyperparameters()
            model = model_initializer(filters, dropout_rate, learning_rate)

            # Train the model for a few epochs
            model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs_per_trial,
                batch_size=batch_size,
                shuffle=False,
                verbose=2
            )

            # Log memory usage
            print(f"Congrats you actually fitted a model")

            # Record results (validation loss or accuracy)
            val_loss = model.evaluate(val_generator, verbose=0)
            results.append((filters, dropout_rate, learning_rate, val_loss))

            # Cleanup to free memory
            keras.backend.clear_session()

        # Move to the next stage
        # Reduce the number of trials, increase epochs
        num_trials = num_trials // 2
        epochs_per_trial = min(epochs_per_trial * 2, max_epochs)

    return results

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    timestamps = generate_timestamps(
        start=dt.datetime(2024, 11, 6, 0, 0, tzinfo=dt.UTC),
        end=dt.datetime(2024, 11, 6, 12, 0, tzinfo=dt.UTC),
    )

    rng = rnd.default_rng(seed=42)
    rng.shuffle(timestamps)
    t_train, t_val = timestamps[:10], timestamps[10:]
    t_test = generate_timestamps(
        start=dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.UTC),
        end=dt.datetime.now(tz=dt.UTC),
    )

    print("Length of train dataset: ", len(t_train))
    print("Length of val dataset: ", len(t_val))
    print("Length of test dataset: ", len(t_test))

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

    # print("About to run the hyperparameter tuning loop")
    # hyperparam_results = run_hyperparameter_tuning(train_generator, val_generator)

    # best_params = sorted(hyperparam_results, key=lambda x : x[3])[0]
    # print(f"Best Hyperparams = {best_params}")
    filters = 16
    dropout_rate = .35
    learning_rate = 1e-3
    model = model_initializer(filters, dropout_rate, learning_rate)
    print(f"Model Made")
    model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=2,
                batch_size=1,
                shuffle=False,
                verbose=2
            )
    print("Model Trained REALLY")

    # best_model.fit(
    #     train_generator, epochs=10, checkpoint_callback=[checkpoint_callback]
    # )
    # final_loss, final_mae = best_model.evaluate(X_test, y_test)
    # print(f"Test Loss: {final_loss}, Test MAE: {final_mae}")
