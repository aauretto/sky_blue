from Logger import LOGGER
import datetime as dt
import multiprocessing

import keras
import keras_tuner as kt
from keras.callbacks import ModelCheckpoint
from numpy import random as rnd
from goes2go import GOES
import os
import pickle
import argparse

import consts as consts
from generator import Generator

CHECKPOINT_DIR = './checkpoints'

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

def build_tuning_model(
    hp: kt.HyperParameters,
    dim_frames=9,
    dim_lat=consts.GRID_RANGE["LAT"],
    dim_lon=consts.GRID_RANGE["LON"],
    dim_alt=consts.GRID_RANGE["ALT"],
    dim_bands=6,
):
    """
    Wrapper for build_model that can be used in keras hyperparameter tuning
    """
    hp_filters = hp.Choice("filters", values=[8, 16, 32])
    hp_dropout = hp.Float("dropout", 0.2, 0.5)
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-4, 1e-3, 1e-2])
    return build_model(hp_filters, hp_dropout, hp_learning_rate, dim_frames, dim_lat, dim_lon, dim_alt, dim_bands)
    

def build_model(
    hp_filters,
    hp_dropout,
    hp_learning_rate,
    dim_frames=9,
    dim_lat=consts.GRID_RANGE["LAT"],
    dim_lon=consts.GRID_RANGE["LON"],
    dim_alt=consts.GRID_RANGE["ALT"],
    dim_bands=6,
):
    """
    Builds the model architecture
    """
    model = keras.Sequential()
    model.add(keras.layers.Input((dim_frames, dim_lat, dim_lon, dim_bands)))

    # 2D Convolutional LSTM layer
    model.add(
        keras.layers.ConvLSTM2D(
            filters=hp_filters,
            kernel_size=(3, 3), # Todo may need to become 5x5
            padding="same",
            return_sequences=True,
        )
    )

    # Dropout layer for regularization
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
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError()],
    )

    return model

def make_train_val_gens(start=dt.datetime(2024, 11, 6, 0, 0, tzinfo=dt.UTC),
                        end=dt.datetime(2024, 11, 6, 12, 0, tzinfo=dt.UTC),
                        seed=42):
    """
    Generates timestamps for the given range

    Return:
        gen_train: the Generator of the train set data
        gen_val: the Generator of the val set data
    """
    rng = rnd.default_rng(seed=seed)
    # Generate dataset timestamps
    timestamps = generate_timestamps(
        start=start,
        end=end,
    )
    rng.shuffle(timestamps)
    t_train, t_val = timestamps[:10], timestamps[10:]

    sat = GOES(satellite=16, product="ABI", domain="C")
    bands = [8, 9, 10, 13, 14, 15]

    # Create the generators
    gen_train = Generator(
        t_train, batch_size=1, frame_size=9, sat=sat, bands=bands
    )  # xs shape: (4,1500,2500, 6) ys shape: (4,1500,2500, 14)
    gen_val = Generator(
        t_val, batch_size=1, frame_size=9, sat=sat, bands=bands
    )  # xs shape: (4,1500,2500, 6) ys shape: (4,1500,2500, 14)

    return gen_train, gen_val

def hyperparam_tune(gen_train, gen_val, tuner_epochs, max_epochs = 7, factor = 10):
    """
    Builds a Keras Tuner, conducts a hyperparameter search, and returns the best model
    to be trained

    Returns:
        model: the best model to be trained
        hp: the best hyperparameters with which a model should be trained
    """
    tuner = kt.Hyperband(
            build_tuning_model,
            objective="val_loss",
            max_epochs=max_epochs, # was 10
            factor=factor, # Was 3
            directory="tuning",
            project_name="turbulent",
        )

    # Run the hyperparameter search
    tuner.search(gen_train, validation_data=gen_val, epochs=tuner_epochs)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters()[0]
    print(f"Best hyperparameters are{best_hps}")

    model = tuner.hypermodel.build(best_hps)
    return model, best_hps


def train_model(train_gen, val_gen, model, total_epochs=5):
    """
    Fits the model to the provided dataset Generators, returns the history of the fit model

    Returns:
        history: History object #TODO ?
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model.{epoch:02d}-{val_loss:.2f}.keras')
    
    # Setup checkpoint callback to save full model
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,  # Save full model
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    history = model.fit(gen_train, validation_data=gen_val, epochs=total_epochs, callbacks=[checkpoint_callback])
    return history


def parse_args():
    """
    Command line argument parsing for script usage
    """
    # Create the main parser
    parser = argparse.ArgumentParser(description="Choose the operation mode and relevant arguments.")

    parser.add_argument(
        '--save_path', 
        type=str, 
        required=True, 
        help="The path to save the tuned model that has to be fit. Save path must not have a file extension"
    )

    # Define subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

    # Subparser for 'tune'
    tune_parser = subparsers.add_parser('tune', help="Enable hyperparameter tuning")

    # Subparser for 'ex_machina'
    ex_machina_parser = subparsers.add_parser('ex_machina', help="Run ex machina mode")
    ex_machina_parser.add_argument('filters', type=int, help="The number of filters used by the CONV LSTM layer")
    ex_machina_parser.add_argument('dropout', type=float, help="The dropout rate of the dropout layer.")
    ex_machina_parser.add_argument('learning_rate', type=float, help="The learning rate of the model.")

    # Subparser for 'checkpoint'
    checkpoint_parser = subparsers.add_parser('checkpoint', help="Checkpoint mode")
    checkpoint_parser.add_argument('checkpoint_path', type=str, help="The path to the .keras model checkpoint file")

    # Parse the arguments
    return parser.parse_args()

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    args = parse_args()
    print(args.save_path)
    gen_train, gen_val = make_train_val_gens() # Using default params

    if args.mode == "ex_machina":
        print(f"Running ex machina with arguments: {args.filters}, {args.dropout}, {args.learning_rate}")
        model = build_model(args.filters, args.dropout, args.learning_rate)
    elif args.mode == "tune":
        print("Running hyperparameter tuning...")
        model, best_hps = hyperparam_tune(gen_train, gen_val)
        with open(args.save_path + 'best_hps.pkl') as f:
            pickle.dump(best_hps, f)
    elif args.mode == "checkpoint":
        print(f"Resuming from checkpoint: {args.checkpoint_path}")
        model = keras.models.load_model(args.checkpoint_path)
    history = train_model(gen_train, gen_val, model)
    model.save(args.save_path + '.keras', overwrite=True, include_optimizer=True)
    with open(args.save_path + '.pkl', 'wb') as f:
        pickle.dump(history.history, f)


    # val_loss_per_epoch = history.history["val_loss"]
    # best_epoch = val_loss_per_epoch.index(max(val_loss_per_epoch)) + 1
    # print("Best epoch: %d" % (best_epoch,))
