from astropy.io import fits
from glob import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

DATA_PATH = "/dcs/large/u2288122/Workspace/qpp_classification/consolidated_data/"
DATA_SUB_DIR = "Flare_sims_large/"
FLARES_SUB_DIR = "Sims_noise/"

if __name__ == "__main__":
    # Read the file for flare IDs and amplitudes
    df = pd.concat([pd.read_csv(file)[[' id', ' amp_scale', ' a_qpp']] \
                for file in glob(os.path.join(DATA_PATH, DATA_SUB_DIR, 'file_info*.txt'))])
    # Clean the columns and generate the ground truth.
    df.columns = [col.strip() for col in df.columns.tolist()]
    df.id = df.id.astype('int32')
    df["ground_truth"] = (df.a_qpp > 0) * 1     # Mark if a QPP exists or not.
    
    # Load the flares from the corresponding fits files.
    data = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        with fits.open(os.path.join(DATA_PATH, DATA_SUB_DIR, FLARES_SUB_DIR, f'flare{int(row.id)}.fits')) as flare:
            data.append([row.id, flare[1].data.flux.tolist(), row.ground_truth, row.a_qpp / row.amp_scale])
    # Record the flare_id, flare, ground truth and the amplitude ratio.

    # Store the dataset
    flares_dataset = pd.DataFrame(data, columns=["flare_id", "X", "y", "amp_ratio"]).set_index('flare_id')
    flares_dataset.to_pickle(os.path.join(DATA_PATH, 'flares_large.pkl'))
 