import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import ot

import data_utils
import distribution_variance


def main():
    risk_data = pd.read_csv("risk_data.csv")
    risk_otu_data = risk_data[data_utils.get_otu_columns(risk_data)]
    risk_distance_matrix = squareform(pdist(risk_otu_data.values, metric='braycurtis'))

    noisy_data = data_utils.create_noisy_data(risk_otu_data)
    noisy_data = data_utils.renormalize_data(noisy_data, otu_only=True)
    noisy_distance_matrix = squareform(pdist(noisy_data.values, metric='braycurtis'))

    risk_data['dataset'] = 'orig'
    risk_data['sample_id'] = risk_data['sample_id'] + '_orig'
    noisy_data['dataset'] = 'noisy'
    noisy_data['sample_id'] = noisy_data['sample_id'] + '_noisy'
    noisy_data['dataset'] = 'noisy'
    combined_data = pd.concat([risk_data, noisy_data])
    combined_data.set_index('sample_id', inplace=True)
    distribution_variance.show_variance(combined_data, 'dataset')

    # mucosalibd_data = pd.read_csv("mucosalibd_data.csv")
    # mucosalibd_otu_data = mucosalibd_data[[c for c in mucosalibd_data.columns if c.isnumeric()]]
    # mucosalibd_distance_matrix = squareform(pdist(mucosalibd_otu_data.values, metric='braycurtis'))

    coupling, log = ot.gromov.gromov_wasserstein(risk_distance_matrix, noisy_distance_matrix, verbose=True, log=True)
    gw_distance = log['gw_dist']
    pass


if __name__ == "__main__":
    main()

