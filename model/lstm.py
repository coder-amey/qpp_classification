# External imports
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing import sequence

# Internal imports and global variables
RANDOM_SEED = 47
DATA_PATH = "/dcs/large/u2288122/Workspace/qpp_classification/consolidated_data"
DATA_FILE = "flares.pkl"
N_TIMESTEPS = 300
N_FEATURES = 1
N_EPOCHS = 16
BATCH_SIZE = 32
'''from experiments.datasets import get_coordinate_trajectory_dataset
from experiments.trainer import train_with_spatial_upsample, test_with_spatial_upsample
from global_config.global_config import (
    NUM_CAMERAS,
    CROSS_VALIDATION_COORDINATE_TRAJECTORIES_PATH,
    CROSS_VALIDATION_WHERE_LSTM_PATH,
    CROSS_VALIDATION_MODELS_PATH_WHERE,
    CROSS_VALIDATION_DEPARTURE_CAMERAS_PATH,
    CROSS_VALIDATION_WHERE_TARGETS_PATH,
    NUM_CROSS_VAL_FOLDS,
    FUTURE_TRAJECTORY_LENGTH,
    BASE_HEATMAP_SIZE,
)'''
tf.random.set_seed(RANDOM_SEED) # Random seed for reproducibility

def load_dataset(file_path):
    dataset = pd.read_pickle(file_path)
    X = np.expand_dims(np.array(dataset.X.tolist()), -1)
    y = np.expand_dims(np.array(dataset.y.tolist()), axis=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM_SEED)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load te dataset
    X_train, X_test, y_train, y_test = load_dataset(os.path.join(DATA_PATH, DATA_FILE))
    # Define the model
    model = Sequential()
    model.add(LSTM(units=128, input_shape=(N_TIMESTEPS, N_FEATURES), activation="tanh", recurrent_activation="sigmoid"))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # View and train the model
    print(model.summary())
    model.fit(X_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE)

    # Evaluate
    scores = model.evaluate(X_test, y_test)
    print("Accuracy: %.2f%%" % (scores[1]*100))
