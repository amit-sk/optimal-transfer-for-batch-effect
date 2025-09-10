import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import ot

import data_utils


def main():
    risk_data = pd.read_csv("risk_data.csv")
    risk_otu_data = risk_data[[c for c in risk_data.columns if c.isnumeric()]]
    risk_distance_matrix = squareform(pdist(risk_otu_data.values, metric='braycurtis'))

    noisy_data = data_utils.create_noisy_data(risk_otu_data)
    noisy_data = data_utils.renormalize_data(noisy_data, otu_only=True)
    noisy_distance_matrix = squareform(pdist(noisy_data.values, metric='braycurtis'))

    # mucosalibd_data = pd.read_csv("mucosalibd_data.csv")
    # mucosalibd_otu_data = mucosalibd_data[[c for c in mucosalibd_data.columns if c.isnumeric()]]
    # mucosalibd_distance_matrix = squareform(pdist(mucosalibd_otu_data.values, metric='braycurtis'))

    coupling, log = ot.gromov.gromov_wasserstein(risk_distance_matrix, noisy_distance_matrix, log=True)
    pass


if __name__ == "__main__":
    main()

