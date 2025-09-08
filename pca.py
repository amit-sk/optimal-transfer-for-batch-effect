import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

def main():
    data = pd.read_csv("risk_data.csv")
    otu_data = data[[c for c in data.columns if c.isnumeric()]]
    distance_matrix = squareform(pdist(otu_data.values, metric='braycurtis'))

    # X_scaled = StandardScaler().fit_transform(otu_data.values)
    # emb = PCA(n_components=2).fit_transform(X_scaled)

    mod = MDS(n_components=2, dissimilarity="precomputed").fit_transform(distance_matrix)

    for group in np.unique(data.phenotype):
        idx = (data.phenotype == group)
        # plt.scatter(emb[idx, 0], emb[idx, 1], label=group)#, alpha=0.75)
        plt.scatter(mod[idx, 0], mod[idx, 1], label=group)#, alpha=0.75)

    plt.legend(title="phenotype")
    plt.title("PCA of OTU Relative Abundance")
    plt.show()


if __name__ == "__main__":
    main()

