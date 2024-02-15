'''
TO-DO:
1) 3x small sets
2) Tune params (w_size, b_size)
3) 3 models * 3 sets
========================================
4) Increase w_size and make 1 large
5) Train model on large
6) README about w_size, img_size, epochs, b_size
'''


import numpy as np
import pandas as pd
import os

from model.CNN_classifier import CNN
from preprocessing.flare2wavelet import flare2wavelet

MODE = "Evaluate" # "Evaluate", "Explore" or "Match"
MODEL = "QPP_detector_300t_ws50.ml"
DATASETS = ["wavelets_ws20.pkl", "wavelets_ws30.pkl", "wavelets_ws40.pkl", "wavelets_ws50.pkl"]
DATA_PATH = "/dcs/large/u2288122/Workspace/qpp_classification/consolidated_data"
def set_dims(x):
    x = np.expand_dims(np.array(x), axis=-1)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    return x


model = CNN.load(MODEL)
datasets = [pd.read_pickle(os.path.join(DATA_PATH, pickle)) for pickle in DATASETS]

if MODE == "Explore":
    dataset = datasets[0]
    # Test random samples
    '''
    df = pd.concat([dataset[dataset.y == 1].sample(n=5), dataset[dataset.y == 0].sample(n=5)])
    df = df.sample(frac=1).reset_index(drop=True)
    '''
    #OR
    # Test individual examples
    indices = [157421, 214510, 253143]
    df = dataset.loc[indices]

    window_sizes = [30] * len(df)
    print("pred\ttruth")
    for w_size, (index, row) in zip(window_sizes, df.iterrows()):
        _, x = flare2wavelet(row.flare, window_size=w_size, plot=True)
        print(np.squeeze(
            model.predict(set_dims(x)).numpy()), row.y, sep="\t")

if MODE == "Evaluate":
# Test the whole dataset
    for dataset in datasets:
        print(f"{MODEL} model evaluation:")
        model.evaluate(set_dims(dataset.X.tolist()), set_dims(dataset.y.tolist()))

if MODE == "Match":
    dataset = datasets[0]
    y_pred = model.predict(set_dims(dataset.X.tolist())).numpy()

    print("pred\ttruth\tmatch")
    for y_, y in zip(np.squeeze(y_pred), dataset.y.tolist()):
        print(int(y_), y, (y == int(y_)), sep="\t")

    mismatch_indices = dataset[dataset.y != np.squeeze(y_pred)].index.tolist()
    print(f"Mismatch indices: {mismatch_indices}")

"""
                Train            Val            Test        (loss/acc/auc)
Model
ws20    .21/.96/.98     .6/.72/.86      1.4/.65/.85
ws30    .09/.96/.99     .09/.94/1       2.7/.6/.66
ws40    .12/.98/.99     .71/.77/.79     3.2/.65/.70
ws50    .05/.99/.999    .52/.89/.91     2.3/.60/.786

           ws_20            ws_30            ws_40        (loss/acc/auc)
Model
ws20    .42/.88/.95     1.71/.75/.83    4.2/.64/.72
ws30    .79/.75/.84     .59/.91/.94     .81/.85/.92
ws40    1.1/.74/.79     .84/.80/.90     .74/.90/.93
ws50    1.49/.69/.76    1.2/.77/.84     .87/.83/.90

For small dataset, ws_30 model is most consistent. Results do vary by window_size.
Larger batch size leads to faster convergence.
"""