import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from skbio.diversity import beta_diversity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from skbio.stats.distance import permanova


SEED = 1


def get_permanova_results(data, group_col):
    distance_matrix = beta_diversity(metric="braycurtis", counts=data.values, ids=data.index)
    permanova_results = permanova(distance_matrix, group_col)
    return permanova_results


def pcoa(data, group_col, seed=SEED):
    distance_matrix = squareform(pdist(data.values, metric='braycurtis'))
    mod = MDS(n_components=2, dissimilarity="precomputed", random_state=seed).fit_transform(distance_matrix)

    for group in np.unique(group_col):
        idx = (group_col == group)
        plt.scatter(mod[idx, 0], mod[idx, 1], label=group, alpha=0.75)

    plt.legend(title=group_col.name)
    plt.title("PCoA of OTU Relative Abundance")
    plt.show()


def show_variance(data, group_col_name):
    otu_data = data[[c for c in data.columns if c.isnumeric()]]
    permanova_results = get_permanova_results(otu_data, data[group_col_name])
    print(f"PERMANOVA results:\n{permanova_results}\n")

    pcoa(otu_data, data[group_col_name])


def main():
    # risk data
    print("RISK data:")
    risk_data = pd.read_csv("risk_data.csv")
    show_variance(risk_data, 'phenotype')

    """
    PCA:
    # X_scaled = StandardScaler().fit_transform(otu_data.values)
    # emb = PCA(n_components=2).fit_transform(X_scaled)
    """

    # mucosalibd data
    print("MucosalIBD data:")
    mucosalibd_data = pd.read_csv("mucosalibd_data.csv")
    show_variance(mucosalibd_data, 'phenotype')

    print("PERMANOVA between datasets:")
    risk_data['dataset'] = 'RISK'
    mucosalibd_data['dataset'] = 'MucosalIBD'
    combined_data = pd.concat([risk_data, mucosalibd_data])
    combined_data.fillna(0.0, inplace=True)
    combined_data.set_index('sample_id', inplace=True)

    show_variance(combined_data, 'dataset')
    show_variance(combined_data, 'phenotype')
    
    print("Done.")



if __name__ == "__main__":
    main()

