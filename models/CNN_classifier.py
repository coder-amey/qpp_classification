# External imports
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D
from tensorflow.keras.utils.vis_utils import plot_model

# Internal imports and global variables
RANDOM_SEED = 47
DATA_PATH = "/dcs/large/u2288122/Workspace/qpp_classification/consolidated_data"
MODEL_PATH = "/dcs/large/u2288122/Workspace/qpp_classification/saved_models" 
DATA_FILE = "wavelets.pkl"
IMG_SIZE = (300, 29)
N_EPOCHS = 16
BATCH_SIZE = 4
tf.random.set_seed(RANDOM_SEED) # Random seed for reproducibility

def load_dataset(file_path):
    dataset = pd.read_pickle(file_path)
    X = np.expand_dims(np.array(dataset.X.tolist()), axis=-1)
    y = np.expand_dims(np.array(dataset.y.tolist()), axis=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)
    return X_train, y_train, X_test, y_test

class CNN:
    def __init__(self, model=None):
        if model is None:
            self.model = Sequential([
                Conv2D(
                    64,
                    (12, 4),
                    strides=1,
                    activation='relu',
                    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)
                ),
                MaxPooling2D(7),
                Conv2D(32, 7, strides=1, activation='relu'),
                MaxPooling2D((5)),
                Conv2D(16, 5, strides=1, activation='relu'),
                MaxPooling2D((3)),
                Dense(256, activation='relu'),
                Dropout((0.125)),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid'),
            ])
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'f1_score'])
        
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
    
    def train(self, X_train, y_train, plot_arch=False):
        # Summarize layers
        print(f"Model summary:\n{self.model.summary()}")
        if plot_arch:
            plot_model(self.model, to_file="model.png", show_shapes=True)
        return self.model.fit(X_train, y_train, epochs=N_EPOCHS, batch_size=BATCH_SIZE)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        scores = self.model.evaluate(X_test, y_test)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        print("F1-scoer: %.2f%%" % (scores[2]))


if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset(os.path.join(DATA_PATH, DATA_FILE))
    print(f"Dataset dimensions:\nTrain\t\t\tTest")
    for series in dataset:
        print(series.shape, end=" ")

    X_train, y_train, X_test, y_test = dataset
    
    detector = CNN()
    detector.train(X_train=X_train, y_train=y_train, plot_arch=True)
    detector.evaluate(X_test=X_test, y_test=y_test)