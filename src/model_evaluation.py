import pickle
import keras
import matplotlib.pyplot as plt
import satellite as st
import datetime as dt
import random as rnd
from cacheReading import read_pirep_cache, retrieve_from_pirep_cache
from consts import PIREP_RELEVANCE_DURATION
from goes2go import GOES
from generator import Generator

SAVE_DIR = '/cluster/tufts/capstone25skyblue/models/2024_02_02-05_ex_machina_defaults/'
HISTORY_SAVE_PATH = SAVE_DIR + 'simon_2024_02_02-052025_04_14_20_03history.pkl'
MODEL_SAVE_PATH = SAVE_DIR + 'simon_2024_02_02-052025_04_14_20_03model.keras'


def visualize_history(history, image_save = None):
    """
    Given a history dict saved from the model training process, visualizes the
    Mean Absolute Error on the validation dataset and as well as visualizes the
	

    Parameters
    ----------
    history: dict
        The history dict that is loaded afterbeing saved in model.py
    imag_save: str | None
        Either saves the image to the specified path or shows it

    Returns
    -------
    None

    """
    val_loss = history['val_loss']
    val_mae = history['val_mean_absolute_error']
    epochs = [i + 1 for i in range(len(val_loss))]
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_loss = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss', color=color_loss)
    ax1.plot(epochs, val_loss, color=color_loss, label='val_loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_loss)

    ax2 = ax1.twinx()
    color_mae = 'tab:blue'
    ax2.set_ylabel('Validation MAE', color=color_mae)
    ax2.plot(epochs, val_mae, color=color_mae, label='val_mae', linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color_mae)

    plt.xticks(epochs)

    fig.suptitle('Validation Metrics Over Epochs', fontsize=14)
    fig.tight_layout()
    if image_save:
        plt.savefig(f'/skyblue/{image_save}')
    else:
        plt.show()

def make_test_gen(start: dt.datetime,
                  end: dt.datetime,
                  batch_size: int = 1):
    """
    Generates timestamps for the given range

    Inputs:
        start: dt.datetime
            for the start of the range
        end: dt.datetime
            the end of the range
        batch_size: int 
            size of the batches in which timestamps should be passed to Generator

    Return:
        gen_test:tf.keras.utils.Sequence or generator
            the Generator of the test set data
    """

    # Generate dataset timestamps
    timestamps = st.generate_timestamps(
        start=start,
        end=end,
    )

    
    all_test = [item for sublist in timestamps for item in sublist]

    print(f"The total number of testing timestamps is: {len(all_test)}")
    times, reports = read_pirep_cache('/cluster/tufts/capstone25skyblue/Caches/2025_2_4_12-6_0PIREPpickled.csv')

    delta = dt.timedelta(minutes = PIREP_RELEVANCE_DURATION)
    
    start_train, end_train = retrieve_from_pirep_cache(min(all_test) - delta, max(all_test) + delta, times)
    test_y_times = times[start_train:end_train]
    test_y_reports = reports[start_train:end_train]



    sat = GOES(satellite=16, product="ABI", domain="C")
    bands = [8, 9, 10, 13, 14, 15]

    # Create the generators
    gen_test = Generator(
        timestamps, test_y_times, test_y_reports, batch_size=batch_size, frame_size=9, sat=sat, bands=bands
    ) 


    print(f"The number of test batches is {len(gen_test)}")

    return gen_test



if __name__ == '__main__':
    start = dt.datetime(2025, 2, 5, 0, tzinfo=dt.UTC)
    end = dt.datetime(2025, 2, 5, 9, tzinfo=dt.UTC)
    test_gen = make_test_gen(start, end)

    print("Starting")
    inference_model = keras.models.load_model(MODEL_SAVE_PATH)
    print("About to open the History callback")
    with open(HISTORY_SAVE_PATH, 'rb') as f:
        history = pickle.load(f)
    print("all loaded")
    print(inference_model)
    print(history)


    visualize_history(history, '3_day_history')
    
    # Results print to terminal
    results = inference_model.evaluate(test_gen, return_dict=True)

