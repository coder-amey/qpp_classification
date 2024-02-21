import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from model.CNN_classifier import CNN
from preprocessing.flare2wavelet import flare2wavelet

MODE = "Evaluate" # "Evaluate", "Explore", "Match" or "Plot"
MODEL = "QPP_detector_500t_ws40_optim.ml"
DATASETS = ["wavelets_large_ws30.pkl", "wavelets_large_ws40.pkl", "wavelets_large_ws50.pkl", "wavelets_large_ws60.pkl"]
DATA_PATH = "/dcs/large/u2288122/Workspace/qpp_classification/consolidated_data"

def set_dims(x):
    x = np.expand_dims(np.array(x), axis=-1)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    return x


detector = CNN.load(MODEL)
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
    indices = [74211, 919369, 36048, 518045, 278119]
    df = dataset.loc[indices]

    window_sizes = [40] * len(df)
    if "amp_ratio" in df.columns:
        print("pred\ttruth\tamp_ratio")
        for w_size, (index, row) in zip(window_sizes, df.iterrows()):
            _, x = flare2wavelet(row.flare, window_size=w_size, plot=True)
            print(np.squeeze(
                detector.predict(set_dims(x)).numpy()), row.y, row.amp_ratio, sep="\t")

    else:
        print("pred\ttruth")
        for w_size, (index, row) in zip(window_sizes, df.iterrows()):
            _, x = flare2wavelet(row.flare, window_size=w_size, plot=True)
            print(np.squeeze(
                detector.predict(set_dims(x)).numpy()), row.y, sep="\t")

if MODE == "Evaluate":
# Test the whole dataset
    for dataset in datasets:
        print(f"{MODEL} model evaluation:")
        detector.evaluate(set_dims(dataset.X.tolist()), set_dims(dataset.y.tolist()))

if MODE == "Match":
    dataset = datasets[0]
    y_pred = detector.predict(set_dims(dataset.X.tolist())).numpy()

    print("pred\ttruth\tmatch")
    for y_, y in zip(np.squeeze(y_pred), dataset.y.tolist()):
        print(int(y_), y, (y == int(y_)), sep="\t")

    mismatch_indices = dataset[dataset.y != np.squeeze(y_pred)].index.tolist()
    print(f"Mismatch indices: {mismatch_indices}")
    if "amp_ratio" in df.columns:
        dataset.amp_ratio = dataset.amp_ratio.round(1)
        print(dataset.loc[mismatch_indices, "amp_ratio"].value_counts())

if MODE == "Plot" and detector.logs:
    print("\n\nModel Training Report:\n\tTraining set:", end="\t")
    print("Loss: %.2f" % (detector.logs["train_log"]['loss'][-1]), end="\t")
    print("Accuracy: %.2f%%" % (detector.logs["train_log"]['accuracy'][-1]*100), end="\t")
    print("AUC: %.2f" % (detector.logs["train_log"]['auc'][-1]))

    print("\tValidation set:", end="\t")
    print("Loss: %.2f" % (detector.logs["train_log"]['val_loss'][-1]), end="\t")
    print("Accuracy: %.2f%%" % (detector.logs["train_log"]['val_accuracy'][-1]*100), end="\t")
    print("AUC: %.2f" % (detector.logs["train_log"]['val_auc'][-1]))

    plt.plot(detector.logs["train_log"]['loss'], label='Training Loss')
    if 'val_loss' in detector.logs["train_log"]:
        plt.plot(detector.logs["train_log"]['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()