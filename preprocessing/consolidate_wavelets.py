import numpy as np
import os
import pandas as pd

from flare2wavelet import flare2wavelet

DATA_PATH = "/dcs/large/u2288122/Workspace/qpp_classification/consolidated_data"

if __name__ == "__main__":
    flares_dataset = pd.read_pickle(os.path.join(DATA_PATH, 'flares.pkl'))
    wavelets_dataset = flares_dataset.copy()
    wavelets_dataset = wavelets_dataset.rename(columns={"X": "flare"})
    power_spectrum = []
    padded_coip = []
    padded_shape = (300, 29)

    for flare in wavelets_dataset.flare.tolist():
        power, coi_power = flare2wavelet(flare)
        power_spectrum.append(power)
        # Pad extra zeros to the left and the bottom of the array.
        padding_needed = ((0, padded_shape[0] - coi_power.shape[0]), \
                          (padded_shape[1] - coi_power.shape[1], 0))
        padded_coip.append(np.pad(coi_power, padding_needed, mode="constant"))
    
    wavelets_dataset["power_spectrum"] = power_spectrum
    wavelets_dataset["X"] = padded_coip
    wavelets_dataset = wavelets_dataset[["flare", "power_spectrum", "X", "y"]]
    wavelets_dataset.to_pickle(os.path.join(DATA_PATH, 'wavelets.pkl'))