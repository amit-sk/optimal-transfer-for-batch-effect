import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import ot

def main():
    risk_data = pd.read_csv("risk_data.csv")
    risk_otu_data = risk_data[[c for c in risk_data.columns if c.isnumeric()]]

    risk_distance_matrix = squareform(pdist(risk_otu_data.values, metric='braycurtis'))

    coupling, log = ot.gromov.gromov_wasserstein(risk_distance_matrix, risk_distance_matrix, log=True)
    pass


if __name__ == "__main__":
    main()

