import os
import pandas as pd
import numpy as np


PATH_TO_RISK_DATA = os.path.join('.', 'raw_data','RISK.tsv')
PATH_TO_RISK_METADATA = os.path.join('.','raw_data', 'metadata.txt')


def obtain_data(data, metadata):
    processed_data = pd.DataFrame()

    # iterate over samples
    for idx, _ in data.transpose().iterrows():
        if idx in ['# OTU','taxonomy']:
            continue

        meta = metadata[metadata.sample_accession == idx]
        phenotype = meta.disease.iloc[0]
        if phenotype not in ['control', 'CD']:
            continue

        # process to relative abundance
        sample_data = data[['# OTU', idx]].copy()
        sample_data[idx] = sample_data[idx] / sample_data[idx].sum()

        new_row = {'sample id': idx, 'phenotype': phenotype}
        new_row.update({int(r['# OTU']):r[idx] for _, r in sample_data.iterrows()})
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
    risk_data = pd.read_csv(PATH_TO_RISK_DATA, sep='\t')
    risk_meta = pd.read_csv(PATH_TO_RISK_METADATA, sep='\t')

    data = obtain_data(risk_data, risk_meta)
    data = filter_uncommon_otus(data)
    data.to_csv("data.csv")

