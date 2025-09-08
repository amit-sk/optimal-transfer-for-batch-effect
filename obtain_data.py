import os
import pandas as pd
import numpy as np


PATH_TO_RISK_DATA = os.path.join('.', 'raw_data','RISK.tsv')
PATH_TO_RISK_METADATA = os.path.join('.','raw_data', 'risk_metadata.txt')
PATH_TO_MUCOSALIBD_DATA = os.path.join('.','raw_data', 'MucosalIBD.tsv')
PATH_TO_MUCOSALIBD_METADATA = os.path.join('.','raw_data', 'mucosalibd_metadata.txt')


def obtain_relative_abundance_data(data, metadata):
    processed_data = pd.DataFrame()

    # iterate over samples
    for idx, _ in data.transpose().iterrows():
        if idx in ['# OTU','taxonomy']:
            continue

        sample_meta = metadata[metadata.sample_accession_16S == idx]
        phenotype = sample_meta.disease.iloc[0]
        if phenotype not in ['control', 'CD']:
            continue

        # process to relative abundance
        sample_data = data[['# OTU', idx]].copy()
        sample_data[idx] = sample_data[idx] / sample_data[idx].sum()

        new_row = {'sample id': idx, 'phenotype': phenotype}
        new_row.update({int(r['# OTU']):r[idx] for _, r in sample_data.iterrows()})
        processed_data = pd.concat([processed_data, pd.DataFrame([new_row])], ignore_index=True)

    return processed_data


def filter_uncommon_otus(data, should_appear_in=0.05, min_abundance=0.005):
    otus = [c for c in data.columns if type(c) is int]
    non_otu_columns = [c for c in data.columns if not type(c) is int]
    amount_greater_than_min = (data[otus] >= min_abundance).sum()
    otus_to_keep = amount_greater_than_min[amount_greater_than_min >= (should_appear_in * len(data))].index

    cols = non_otu_columns
    cols.extend(otus_to_keep)
    return data[cols]


def main():
    risk_data = pd.read_csv(PATH_TO_RISK_DATA, sep='\t')
    risk_meta = pd.read_csv(PATH_TO_RISK_METADATA, sep='\t')
    mucosalibd_data = pd.read_csv(PATH_TO_MUCOSALIBD_DATA, sep='\t')
    mucosalibd_meta = pd.read_csv(PATH_TO_MUCOSALIBD_METADATA, sep='\t')

    risk_processed_data = obtain_relative_abundance_data(risk_data, risk_meta)
    risk_processed_data = filter_uncommon_otus(risk_processed_data)
    risk_processed_data.to_csv("risk_data.csv")

    mucosalibd_processed_data = obtain_relative_abundance_data(mucosalibd_data, mucosalibd_meta)
    mucosalibd_processed_data = filter_uncommon_otus(mucosalibd_processed_data)
    mucosalibd_processed_data.to_csv("mucosalibd_data.csv")


if __name__ == "__main__":
    main()

