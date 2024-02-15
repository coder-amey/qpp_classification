# External imports
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.metrics import AUC
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D


# Internal imports and global variables
RANDOM_SEED = 47
DATA_PATH = "/dcs/large/u2288122/Workspace/qpp_classification/consolidated_data"
MODEL_PATH = "/dcs/large/u2288122/Workspace/qpp_classification/model/saved_models"
NAME = False #"QPP_detector_300t_ws30.ml"
DATA_FILE = "wavelets_ws30.pkl"
VALIDATION = True
PLOT = False
IMG_SIZE = (300, 29)
N_EPOCHS = 32
BATCH_SIZE = 16
tf.random.set_seed(RANDOM_SEED) # Random seed for reproducibility

def load_dataset(file_path, val=False):
    dataset = pd.read_pickle(file_path)
    X = np.expand_dims(np.array(dataset.X.tolist()), axis=-1)
    y = np.expand_dims(np.array(dataset.y.tolist()), axis=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    if val:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=RANDOM_SEED)
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    else:
        return X_train, y_train, X_test, y_test

class CNN:
    def __init__(self, model=None):
        if model is None:
            self.model = Sequential([
                Conv2D(
                    32,
                    3,
                    strides=1,
                    activation='relu',
                    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)
                ),
                MaxPooling2D(3),
                Conv2D(16, 3, strides=1, activation='relu'),
                MaxPooling2D(3),
                Flatten(),
                Dense(256, activation='relu'),
                Dropout((0.25)),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
                ])
            
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC()])
        else:
            self.model = model
    
    def __init_subclass__(cls) -> None:
        pass

    def save(self, name="QPP_detector.ml", path=MODEL_PATH):
        self.model.save(os.path.join(path, name))
    
    @staticmethod
    def load(name="QPP_detector.ml", path=MODEL_PATH):
        model = tf.keras.models.load_model(os.path.join(path, name))
        return CNN(model)
    
    def train(self, X_train, y_train, val=None, plot_arch=False):
        # Summarize layers
        print(f"Model summary:\n{self.model.summary()}")
        if plot_arch:
            plot_model(self.model, to_file="model.png", show_shapes=True)
        if val:
            X_val, y_val = val
            return self.model.fit(X_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))
        else:
            return self.model.fit(X_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE)
    
    def predict(self, X):
        return tf.round(self.model.predict(X))
    
    def evaluate(self, X_test, y_test):
        scores = self.model.evaluate(X_test, y_test)
        print("\n\nModel Evaluation Report:")
        print("Loss: %.2f" % (scores[0]))
        print("Accuracy: %.2f%%" % (scores[1]*100))
        print("AUC: %.2f" % (scores[2]))


if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset(os.path.join(DATA_PATH, DATA_FILE), val=VALIDATION)
    detector = CNN()
    logs = None

    if VALIDATION:
        print(f"Dataset dimensions:\nTrain\t\t\t\tVal\t\t\t\tTest")
        for series in dataset:
            print(series.shape, end=" ")

        X_train, y_train, X_val, y_val, X_test, y_test = dataset
        print(f"\nBinary data balance:\
              \nTrain: {np.sum(y_train, axis=0) / y_train.shape[0]}\
              \tVal: {np.sum(y_val, axis=0) / y_val.shape[0]}\
              \tTrain: {np.sum(y_test, axis=0) / y_test.shape[0]}")
        logs = detector.train(X_train=X_train, y_train=y_train, val=[X_val, y_val], plot_arch=PLOT)
    else:
        print(f"Dataset dimensions:\nTrain\t\tTest")
        for series in dataset:
            print(series.shape, end=" ")

        X_train, y_train, X_test, y_test = dataset
        print(f"\nBinary data balance:\
              \nTrain: {np.sum(y_train, axis=0) / y_train.shape[0]}\
              \tTrain: {np.sum(y_test, axis=0) / y_test.shape[0]}")
        logs = detector.train(X_train=X_train, y_train=y_train, plot_arch=PLOT)
    
    
    detector.evaluate(X_test=X_test, y_test=y_test)
    plt.plot(logs.history['loss'], label='Training Loss')
    if VALIDATION:
        plt.plot(logs.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    if NAME:
        detector.save(name=NAME)
