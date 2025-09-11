import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import ot

import data_utils
import distribution_variance


def barycentric_projection(coupling, src_dataset, x_onto_y=True):
    """ Based on code from SCOTv1 """
    if x_onto_y:
        # Projecting the first domain onto the second domain
        weights = np.sum(coupling, axis=1)
        src_aligned = np.matmul(coupling, src_dataset) / weights[:, None]
    else:
        # Projecting the second domain onto the first domain
        weights = np.sum(coupling, axis=0)
        src_aligned = np.matmul(np.transpose(coupling), src_dataset) / weights[:, None]

    return src_aligned


def main():
    risk_data = pd.read_csv("risk_data.csv")
    risk_otu_data = risk_data[data_utils.get_otu_columns(risk_data)]
    risk_distance_matrix = squareform(pdist(risk_otu_data.values, metric='braycurtis'))

    noisy_data = data_utils.create_noisy_data(risk_data, proportion_of_std=0.05)
    noisy_data = data_utils.renormalize_data(noisy_data)
    noisy_otu_data = noisy_data[data_utils.get_otu_columns(noisy_data)]
    noisy_distance_matrix = squareform(pdist(noisy_otu_data.values, metric='braycurtis'))

    risk_data['dataset'] = 'orig'
    risk_data['sample_id'] = risk_data['sample_id'] + '_orig'
    noisy_data['dataset'] = 'noisy'
    noisy_data['sample_id'] = noisy_data['sample_id'] + '_noisy'
    noisy_data['dataset'] = 'noisy'
    combined_data = pd.concat([risk_data, noisy_data])
    combined_data.set_index('sample_id', inplace=True)
    indexes = combined_data.index
    pairs = [(indexes.get_loc(i), indexes.get_loc(i.replace('_orig','_noisy'))) for i in indexes if i.endswith('_orig')]
    distribution_variance.show_variance(combined_data, 'dataset', pcoa_pairs=pairs)
    fracs = distribution_variance.calc_domain_avg_FOSCTTM(risk_otu_data.values, noisy_otu_data.values)

    # mucosalibd_data = pd.read_csv("mucosalibd_data.csv")
    # mucosalibd_otu_data = mucosalibd_data[data_utils.get_otu_columns(mucosalibd_data)]
    # mucosalibd_distance_matrix = squareform(pdist(mucosalibd_otu_data.values, metric='braycurtis'))

    coupling, log = ot.gromov.gromov_wasserstein(risk_distance_matrix, noisy_distance_matrix, verbose=True, log=True)
    gw_distance = log['gw_dist']
    print(f'GW distance: {gw_distance}')

    projected = barycentric_projection(coupling, noisy_otu_data.values, x_onto_y=False)
    fracs = distribution_variance.calc_domain_avg_FOSCTTM(risk_otu_data.values, projected)
    
    # TODO: create combined data with projected data and show variance
    # distribution_variance.show_variance(combined_data, 'dataset', pcoa_pairs=pairs)
    pass


if __name__ == "__main__":
    main()

