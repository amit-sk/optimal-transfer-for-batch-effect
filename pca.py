import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    data = pd.read_csv("data.csv")
    otu_data = data[[c for c in data.columns if c.isnumeric()]]
    X_scaled = StandardScaler().fit_transform(otu_data.values)
    emb = PCA(n_components=2).fit_transform(X_scaled)

    for group in np.unique(data.phenotype):
        idx = (data.phenotype == group)
        plt.scatter(emb[idx, 0], emb[idx, 1], label=group)#, alpha=0.75, s=28)

    plt.legend(title="phenotype")
    plt.show()


if __name__ == "__main__":
    main()

