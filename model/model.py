from tensorflow import keras
from keras.layers import Dense

# Input: some kind of time series

model = keras.Sequential(
    [
        Dense(
            128, activation="relu", input_shape=((n, 1500, 2500),)
        ),  # Input layer and first hidden layer
        Dense(64, activation="relu"),  # Second hidden layer
        Dense(
            2, activation="softmax"
        ),  # Output layer # Binary class for yes no turbulence
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.summary()

model.fit(X, y, epochs=..., batch_size=...)


# model.evaluate(): To calculate the loss values for the input data
# model.predict(): To generate network output for the input data
