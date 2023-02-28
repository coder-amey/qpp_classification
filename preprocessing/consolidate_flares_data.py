from astropy.io import fits
import numpy as np
import os
import pandas as pd

DATA_PATH = "/dcs/large/u2288122/Workspace/qpp_classification/consolidated_data"
DATA_SUB_DIR = ["HH1", "HH2", "HH3"]
DATA_FILES = ["HH1/consolidated_qpp_type_hh1.csv", "HH2/consolidated_qpp_type_hh2.csv", "HH3/consolidated_qpp_type_hh3.csv"]

if __name__ == "__main__":
    rows = []
    for csv_file, sub_dir in zip(DATA_FILES, DATA_SUB_DIR):
        df = pd.read_csv(os.path.join(DATA_PATH, csv_file)).astype(int).set_index('flare_id')
        for id in df.index.tolist():
            with fits.open(os.path.join(DATA_PATH, sub_dir, f'flare{id}.fits')) as data:
                rows.append([id, data[1].data.flux.tolist(), df.loc[id, "ground_truth"]])
    flares_dataset = pd.DataFrame(rows, columns=["flare_id", "X", "y"]).set_index('flare_id')
    flares_dataset.to_pickle(os.path.join(DATA_PATH, 'flares.pkl'))
