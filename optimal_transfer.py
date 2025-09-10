import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import ot


SEED = 1


def create_noisy_data(data, proportion_of_std=0.1, seed=SEED):
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


def main():
    risk_data = pd.read_csv("risk_data.csv")
    risk_otu_data = risk_data[[c for c in risk_data.columns if c.isnumeric()]]
    risk_distance_matrix = squareform(pdist(risk_otu_data.values, metric='braycurtis'))

    noisy_data = create_noisy_data(risk_otu_data)
    noisy_distance_matrix = squareform(pdist(noisy_data.values, metric='braycurtis'))

    # mucosalibd_data = pd.read_csv("mucosalibd_data.csv")
    # mucosalibd_otu_data = mucosalibd_data[[c for c in mucosalibd_data.columns if c.isnumeric()]]
    # mucosalibd_distance_matrix = squareform(pdist(mucosalibd_otu_data.values, metric='braycurtis'))

    coupling, log = ot.gromov.gromov_wasserstein(risk_distance_matrix, noisy_distance_matrix, log=True)
    pass


if __name__ == "__main__":
    main()

