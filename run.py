import numpy as np
import pandas as pd
import os

from model.CNN_classifier import CNN

DATA_PATH = "/dcs/large/u2288122/Workspace/qpp_classification/consolidated_data"
def set_dims(x):
    x = np.expand_dims(np.array(x), axis=-1)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    return x


model = CNN.load("QPP_detector_mini.ml")
dataset = pd.read_pickle(os.path.join(DATA_PATH, 'wavelets.pkl'))

# Test random/individual examples
"""
df = pd.concat([dataset[dataset.y == 1].sample(n=5), dataset[dataset.y == 0].sample(n=5)])
df = df.sample(frac=1).reset_index(drop=True)
print("pred\ttruth")
for index, flare in df.iterrows():
    print(model.predict(set_dims(flare.X)).numpy(), set_dims(flare.y), sep="\t")
"""

# Test the whole dataset
y_pred = model.predict(set_dims(dataset.X.tolist())).numpy()

print("pred\ttruth\tmatch")
for y_, y in zip(np.squeeze(y_pred), dataset.y.tolist()):
    print(int(y_), y, (y == int(y_)), sep="\t")

mismatch_indices = np.where(np.squeeze(y_pred) != dataset.y.to_numpy())
print(f"Mismatch indices: {mismatch_indices[0]}")