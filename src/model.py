from goes2go import GOES

# from sklearn.model_selection import train_test_split
import datetime as dt
import numpy as np
import pandas as pd
import keras
import pirep as pr
import satellite as st
import consts as consts

if __name__ == "__main__":
    # INPUT DATA

    # Initialize satellite and bands
    sat_east = GOES(satellite=16, product="ABI", domain="C")
    bands = [8, 9, 10, 13, 14, 15]

    # Fetch satellite data
    data = st.fetch_range(
        start=dt.datetime(2024, 11, 6, 23, 54),
        end=dt.datetime(2024, 11, 7, 00, 14),
        satellite=sat_east,
    )

    # Project data onto grid
    lats, lons = st.calculate_coordinates(data)
    band_data = st.fetch_bands(data, bands)
    timestamps = pd.to_datetime(band_data.coords["t"])
    data = st.smooth(st.project(lats, lons, band_data.data))

    # TODO: take the timestamp associated with each frame, and choose the relevant PIREPs for each frame before binning them per-frame.
    print(f"Frames: {data.shape[0]}")
    assert data.shape[1:] == (
        consts.GRID_RANGE["LAT"],
        consts.GRID_RANGE["LON"],
        len(bands),
    )

    # INPUT LABELS

    # Retrieve PIREPs
    reports: pd.DataFrame = pr.parse_all(
        pr.fetch(
            pr.url(
                date_s=dt.datetime(2024, 11, 6, 23, 54, 0, tzinfo=dt.timezone.utc),
                date_e=dt.datetime(2024, 11, 7, 0, 14, 0, tzinfo=dt.timezone.utc),
            )
        )
    )

    # Convert reports to grids
    grids = pd.DataFrame(
        {
            "Timestamp": reports["Timestamp"],
            "Grid": reports.apply(pr.compute_grid, axis=1),
        }
    )
    print(
        [
            abs((grids["Timestamp"] - timestamp) / pd.Timedelta(minutes=1))
            for timestamp in timestamps
        ]
    )

    # TODO: Spread the PIREPs here
    # labels = np.zeros(
    #     (
    #         data.shape[0],
    #         consts.GRID_RANGE["LAT"],
    #         consts.GRID_RANGE["LON"],
    #         consts.GRID_RANGE["ALT"],
    #     )
    # )
    labels = np.array(
        [
            np.max(
                grids[(grids["Timestamp"] - timestamp) / pd.Timedelta(minutes=1)][
                    "Grid"
                ].to_numpy(),
                axis=0,
            )
            for timestamp in timestamps
        ]
    )
    print(labels.shape)

    assert labels.shape == (
        data.shape[0],
        consts.GRID_RANGE["LAT"],
        consts.GRID_RANGE["LON"],
        consts.GRID_RANGE["ALT"],  # TODO: Change when the altitude range is modified
    )

    # MODEL TRAINING

    # Input: some kind of time series
    X = data
    y = labels
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.33, random_state=42
    # )
    # Model parameters
    num_classes = 2
    input_shape = (3, 1500, 2500, 6)

    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            # keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            # keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            # keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            # keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            # keras.layers.GlobalAveragePooling2D(),
            # keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()
    print(X.shape)
    print(y.shape)
    model.fit(X, y, epochs=10, batch_size=32)

    # model.evaluate(): To calculate the loss values for the input data
    # model.predict(): To generate network output for the input data
