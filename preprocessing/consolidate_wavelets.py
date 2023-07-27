import numpy as np
import os
import pandas as pd

from flare2wavelet import flare2wavelet

DATA_PATH = "/dcs/large/u2288122/Workspace/qpp_classification/consolidated_data"

if __name__ == "__main__":
    flares_dataset = pd.read_pickle(os.path.join(DATA_PATH, 'flares.pkl'))
    wavelets_dataset = flares_dataset.copy()
    wavelets_dataset.rename(columns={"X": "flare"})
    wavelets_dataset["power_spectrum"] = None
    wavelets_dataset["coi_power_spectrum"] = None

    for flare_id in wavelets_dataset.index():
        flare = wavelets_dataset.loc[flare_id, "flare"]
        power, coi_power = flare2wavelet(flare, plot=False)
        wavelets_dataset.loc[flare_id, "power_spectrum"] = power
        wavelets_dataset.loc[flare_id, "X"] = coi_power

    wavelets_dataset.to_pickle(os.path.join(DATA_PATH, 'wavelets.pkl'))