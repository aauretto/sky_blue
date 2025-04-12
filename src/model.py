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
import satellite as st
from cacheReading import read_pirep_cache, retrieve_from_pirep_cache
from consts import PIREP_RELEVANCE_DURATION

CHECKPOINT_DIR = './checkpoints'

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

def make_train_val_gens(start: dt.datetime,
                        end: dt.datetime,
                        batch_size: int = 1,
                        seed=42):
    """
    Generates timestamps for the given range

    Return:
        gen_train: the Generator of the train set data
        gen_val: the Generator of the val set data
    """
    rng = rnd.default_rng(seed=seed)
    # Generate dataset timestamps
    timestamps = st.generate_timestamps(
        start=start,
        end=end,
    )
    rng.shuffle(timestamps)
    t_train, t_val = timestamps[:10], timestamps[10:]
    
    all_train = [item for sublist in t_train for item in sublist]
    all_val = [item for sublist in t_val for item in sublist]
    print(f"The total number of training timestamps is: {len(all_train)}")
    print(f"The total number of validation timestamps is: {len(all_val)}")
    times, reports = read_pirep_cache()

    delta = dt.timedelta(minutes = PIREP_RELEVANCE_DURATION)
    
    start_train, end_train = retrieve_from_pirep_cache(min(all_train) - delta, max(all_train) + delta, times)
    start_val, end_val = retrieve_from_pirep_cache(min(all_val) - delta, max(all_val) + delta, times)
    train_y_times = times[start_train:end_train]
    train_y_reports = reports[start_train:end_train]
    val_y_times = times[start_val:end_val]
    val_y_reports = reports[start_val:end_val]


    sat = GOES(satellite=16, product="ABI", domain="C")
    bands = [8, 9, 10, 13, 14, 15]

    # Create the generators
    gen_train = Generator(
        t_train, train_y_times, train_y_reports, batch_size=batch_size, frame_size=9, sat=sat, bands=bands
    ) 
    gen_val = Generator(
        t_val, val_y_times, val_y_reports, batch_size=batch_size, frame_size=9, sat=sat, bands=bands
    )

    print(f"The number of train batches is {len(gen_train)}")
    print(f"The number of validation batches is {len(gen_val)}")

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
    history = model.fit(train_gen, validation_data=val_gen, epochs=total_epochs, callbacks=[checkpoint_callback])
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

    parser.add_argument(
        '--start_date',
        type=str,
        default="2017_03_01_00_00",
        help="The datetime which is the beginning of the training data in UTC: YYYY_MM_DD[_hh_mm]"
    )

    parser.add_argument(
        '--end_date',
        type=str,
        default="2024_12_31_23_59",
        help="The datetime which is the beginning of the training data in UTC: YYYY_MM_DD[_hh_mm]"
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help="The batch size for the data generators"
    )

    # Define subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

    # Subparser for 'tune'
    tune_parser = subparsers.add_parser('tune', help="Enable hyperparameter tuning")

    # Subparser for 'ex_machina'
    ex_machina_parser = subparsers.add_parser('ex_machina', help="Run ex machina mode")
    ex_machina_parser.add_argument('--filters', type=int, default=16, help="The number of filters used by the CONV LSTM layer")
    ex_machina_parser.add_argument('--dropout', type=float, default=.2, help="The dropout rate of the dropout layer.")
    ex_machina_parser.add_argument('--learning_rate', type=float, default=1e-3, help="The learning rate of the model.")

    # Subparser for 'checkpoint'
    checkpoint_parser = subparsers.add_parser('checkpoint', help="Checkpoint mode")
    checkpoint_parser.add_argument('--checkpoint_path', type=str, help="The path to the .keras model checkpoint file")

    args = parser.parse_args()

    t = dt.datetime.now().strftime("%Y_%m_%d_%H_%M")

    args.save_path += t
    start_date = parse_dates(args.start_date)
    if not start_date:
        parser.error("start_date must be must be in the form YYYY_MM_DD[_hh_mm]")
    args.start_date = start_date
    end_date = parse_dates(args.end_date)
    if not end_date:
        parser.error("end_date must be must be in the form YYYY_MM_DD[_hh_mm]")
    args.end_date = end_date
        
    # Parse the arguments
    return args

def parse_dates(dateStr: str):
    """
    Parses a string into the YEAR MONTH DATE [MINUTES HOURS] format 
    required for dt.datetime
    Input:
        dateStr: str the string received as input to the script
    Returns:
        dateDict: a dictionary of field (str), value (int) pairs to be turned into 
        a datetime object. 
    """
    dateParts = dateStr.split(sep="_")
    if len(dateParts) not in (3, 5):
        return None
    else:
        datePartsints = [int(dp.lstrip('0')) for dp in dateParts]

        dateDict= {
            'YYYY' : datePartsints[0],
            'MM'   : datePartsints[1],
            'DD'   : datePartsints[2],
            'hh'   : 0,
            'mm'   : 0,
        }

        if len(dateParts) == 5:
            dateDict["hh"] = datePartsints[3]
            dateDict["mm"] = datePartsints[4]
    return dateDict

if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    args = parse_args()
    start_date = dt.datetime(args.start_date['YYYY'], args.start_date['MM'], args.start_date['DD'], args.start_date['hh'], args.start_date['mm'], tzinfo=dt.UTC)
    end_date = dt.datetime(args.end_date['YYYY'], args.end_date['MM'], args.end_date['DD'], args.end_date['hh'], args.end_date['mm'], tzinfo=dt.UTC)
    
    print(f"Creating generators with batch_size={args.batch_size}")
    gen_train, gen_val = make_train_val_gens(start_date, end_date, args.batch_size) # Using default params

    if args.mode == "ex_machina":
        print(f"Running ex machina with arguments: {args.filters}, {args.dropout}, {args.learning_rate}")
        model = build_model(args.filters, args.dropout, args.learning_rate)
    elif args.mode == "tune":
        print("Running hyperparameter tuning")
        model, best_hps = hyperparam_tune(gen_train, gen_val)
        with open(args.save_path + 'best_hps.pkl') as f:
            pickle.dump(best_hps, f)
    elif args.mode == "checkpoint":
        print(f"Resuming from checkpoint: {args.checkpoint_path}")
        model = keras.models.load_model(args.checkpoint_path)
    print("Entering final model training")
    history = train_model(gen_train, gen_val, model)
    print("Finished model training")
    model.save(args.save_path + 'model.keras', overwrite=True, include_optimizer=True)
    print("Model saved")
    with open(args.save_path + 'history.pkl', 'wb') as f:
        pickle.dump(history.history, f)


    # val_loss_per_epoch = history.history["val_loss"]
    # best_epoch = val_loss_per_epoch.index(max(val_loss_per_epoch)) + 1
    # print("Best epoch: %d" % (best_epoch,))
