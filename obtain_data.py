import os
import pandas as pd
import numpy as np


PATH_TO_DATA = os.path.join('.', 'raw_data','RISK.tsv')
PATH_TO_METADATA = os.path.join('.','raw_data', 'metadata.txt')


def obtain_data():
    processed_data = pd.DataFrame()

    for idx, row in risk_data_transpose.iterrows():
        if idx in ['# OTU','taxonomy']:
            continue

        metadata = risk_meta[risk_meta.sample_accession == idx]
        phenotype = metadata.disease.iloc[0]
        if phenotype not in ['control', 'CD']:
            continue

        data = risk_data[['# OTU', idx]].copy()
        data[idx] = data[idx] / data[idx].sum()

        new_row = {'sample id': idx, 'phenotype': phenotype}
        new_row.update({int(r['# OTU']):r[idx] for _, r in data.iterrows()})
        processed_data = pd.concat([processed_data, pd.DataFrame([new_row])], ignore_index=True)

    return processed_data


def filter_uncommon_otus(data, should_appear_in=0.1, min_abundance=0.01):
    otus = [c for c in data.columns if type(c) is int]
    non_otu_columns = [c for c in data.columns if not type(c) is int]
    amount_greater_than_min = (data[otus] >= min_abundance).sum()
    otus_to_keep = amount_greater_than_min[amount_greater_than_min >= (should_appear_in * len(data))].index

    cols = non_otu_columns
    cols.extend(otus_to_keep)
    return data[cols]


if __name__ == "__main__":
    risk_data = pd.read_csv(PATH_TO_DATA, sep='\t')
    risk_meta = pd.read_csv(PATH_TO_METADATA, sep='\t')
    risk_data_transpose = risk_data.transpose()

    data = obtain_data()
    data = filter_uncommon_otus(data)
    data.to_csv("data.csv")

