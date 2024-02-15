import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from flare2wavelet import flare2wavelet

DATA_PATH = "/dcs/large/u2288122/Workspace/qpp_classification/consolidated_data"
FLARES_FILENAME = "flares.pkl"
WAVELETS_FILENAME = "wavelets_ws50.pkl"
WINDOW_SIZE = 50

if __name__ == "__main__":
    flares_dataset = pd.read_pickle(os.path.join(DATA_PATH, FLARES_FILENAME))
    wavelets_dataset = flares_dataset.copy()
    wavelets_dataset = wavelets_dataset.rename(columns={"X": "flare"})
    power_spectrum = []
    coi_power_spectrum = []

    for index, row in tqdm(wavelets_dataset.iterrows()):
        flare = row.flare
        try:
            power, coi_power = flare2wavelet(flare, WINDOW_SIZE)
            power_spectrum.append(power)
            coi_power_spectrum.append(coi_power)
        except AssertionError as e:
            print(f"{e}\nSkipping flare...")
            wavelets_dataset = wavelets_dataset.drop(index)
        
    wavelets_dataset["power_spectrum"] = power_spectrum
    wavelets_dataset["X"] = coi_power_spectrum
    if "amp_ratio" in wavelets_dataset.columns:
        wavelets_dataset = wavelets_dataset[["amp_ratio", "flare", "power_spectrum", "X", "y"]]
    else:
        wavelets_dataset = wavelets_dataset[["flare", "power_spectrum", "X", "y"]]
    wavelets_dataset.to_pickle(os.path.join(DATA_PATH, WAVELETS_FILENAME))
    print(f"Retrieved and stored {wavelets_dataset.shape[0]} flares successfully.")