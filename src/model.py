from Logger import LOGGER
import datetime as dt
import multiprocessing

import keras
import keras_tuner as kt
from numpy import random as rnd
from goes2go import GOES

import consts as consts
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


def build_model(
    hp: kt.HyperParameters,
    dim_frames=9,
    dim_lat=consts.GRID_RANGE["LAT"],
    dim_lon=consts.GRID_RANGE["LON"],
    dim_alt=consts.GRID_RANGE["ALT"],
    dim_bands=6,
):
    model = keras.Sequential()
    model.add(keras.layers.Input((dim_frames, dim_lat, dim_lon, dim_bands)))

    # 2D Convolutional LSTM layer
    hp_filters = hp.Choice("filters", values=[8, 16, 32])
    model.add(
        keras.layers.ConvLSTM2D(
            filters=hp_filters,
            kernel_size=(3, 3), # Todo may need to become 5x5
            padding="same",
            return_sequences=True,
        )
    )

    # Dropout layer for regularization
    hp_dropout = hp.Float("dropout", 0.2, 0.5)
    model.add(keras.layers.Dropout(rate=hp_dropout))

    # 1x1 2D Convolutional layer to reduce feature map
    model.add(
        keras.layers.Conv3D(
            filters=dim_alt,
            kernel_size=(1, 1, 1),
            activation="linear",
            padding="same",
        )
    )

    # Compile the model
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-4, 1e-3, 1e-2])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()],
    )

    return model


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    rng = rnd.default_rng(seed=42)

    # Generate dataset timestamps
    timestamps = generate_timestamps(
        start=dt.datetime(2024, 11, 6, 0, 0, tzinfo=dt.UTC),
        end=dt.datetime(2024, 11, 6, 12, 0, tzinfo=dt.UTC),
    )
    rng.shuffle(timestamps)
    t_train, t_val = timestamps[:10], timestamps[10:]
    t_test = generate_timestamps(
        start=dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.UTC),
        end=dt.datetime.now(tz=dt.UTC),
    )

    print(f"Timestamps: {len(t_train)}/{len(t_val)}:{len(t_test)}")

    sat = GOES(satellite=16, product="ABI", domain="C")
    bands = [8, 9, 10, 13, 14, 15]

    # Create the generators
    gen_train = Generator(
        t_train, batch_size=1, frame_size=9, sat=sat, bands=bands
    )  # xs shape: (4,1500,2500, 6) ys shape: (4,1500,2500, 14)
    gen_val = Generator(
        t_val, batch_size=1, frame_size=9, sat=sat, bands=bands
    )  # xs shape: (4,1500,2500, 6) ys shape: (4,1500,2500, 14)
    gen_test = Generator(
        t_test, batch_size=1, frame_size=9, sat=sat, bands=bands
    )  # xs shape: (4,1500,2500, 6) ys shape: (4,1500,2500, 14)

    print(f"Generators: {len(gen_train)}/{len(gen_val)}:{len(gen_test)}")

    # Tune the model
    tuner = kt.Hyperband(
        build_model,
        objective="val_loss",
        max_epochs=7, # was 10
        factor=10, # Was 3
        directory="tuning",
        project_name="turbulent",
    )

    # Run the hyperparameter search
    tuner.search(gen_train, validation_data=gen_val, epochs=10)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps)

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(gen_train, validation_data=gen_val, epochs=50)

    val_loss_per_epoch = history.history["val_loss"]
    best_epoch = val_loss_per_epoch.index(max(val_loss_per_epoch)) + 1
    print("Best epoch: %d" % (best_epoch,))
