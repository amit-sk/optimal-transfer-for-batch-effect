import pandas as pd
import numpy as np


PROJECT_SEED = 1


def renormalize_data(data, otu_only=False):
    if otu_only:
        sums = data.sum(axis=1) 
        return data.div(sums, axis=0)
    
    copy = data.set_index('sample_id')
    otus = [c for c in copy.columns if type(c) is int]
    sums = copy[otus].sum(axis=1) 
    copy[otus] = copy[otus].div(sums, axis=0)
    return copy


def create_noisy_data(data, proportion_of_std=0.1, seed=PROJECT_SEED):
    np.random.seed(seed)
    stds = data.std()
    copy = data.copy()
    
    for sample_id, row in data.iterrows():
        for idx, cell in row.items():
            if cell == 0.0:
                continue

            std = stds[idx] * proportion_of_std
            noise = np.random.normal(0, std)
            new_val = max(0.0, cell + noise)  # no negative values
            copy.at[sample_id, idx] = new_val

    return copy


def get_otu_columns(data):
    return [c for c in data.columns if (type(c) is int) or (type(c) is str and c.isnumeric())]

