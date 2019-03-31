# Importing modules
import pandas as pd
import random

print("Modules loaded, starting with datasets..")

#-------------------------------------------
# Create functions for processing
#-------------------------------------------

def data_split(dataset):
    split_equal = dataset.sample(frac=0.5)
    training = dataset.drop(split_equal.index, axis=0)
    valid = split_equal.sample(frac=0.5)
    test = split_equal.drop(valid.index, axis=0)
    return training, valid, test

def save_dataset(dataset_name, dataset, data_type):
    dataset.to_csv("Datasets/" + dataset_name + "/" + data_type + ".txt", index=False, header=False, sep="\t")

def save_data(dataset, dataset_name):
    training, validation, test = data_split(dataset)
    save_dataset(dataset_name, training, 'train')
    save_dataset(dataset_name, validation, 'valid')
    save_dataset(dataset_name, test, 'test')

# ------------------------------------------
# HepPh dataset
# ------------------------------------------

print("--> Creating HepPh dataset")

# Load in datasets from datasets folder
hepph_net = pd.read_csv("Datasets/cit-HepPH/original/Cit-HepPh.txt", sep="\t", skiprows=4, index_col=False, names=["startnode","endnode"])
hepph_time = pd.read_csv("Datasets/cit-HepPH/original/cit-HepPh-dates.txt", sep="\t", skiprows=1, index_col=False, names=["startnode","timestamp"])

# Merge datasets on startnode column
hepph_full = hepph_net.merge(hepph_time, how="left" ,on="startnode").dropna()
hepph_ordering = ["startnode","timestamp","endnode"]
hepph_full = hepph_full.loc[:, hepph_ordering]

# Generate negative edges
#hepph_labeled = negative_edge(hepph_full, 'timestamp', limited=False, include_edge_label=True)
#hepph_unlabeled = negative_edge(hepph_full, 'timestamp', limited=False, include_edge_label=False)

# Generate test/training split and save as .txt
save_data(hepph_full, 'Cit-HepPh')

print("--> HepPH is ready..")

# ------------------------------------------
# Bitcoin dataset
# ------------------------------------------

print("--> Creating Bitcoin dataset")

# load in dataset from datasets folder
bitcoin_full = pd.read_csv("Datasets/BitcoinSign/original/soc-sign-bitcoinotc.csv", index_col=False, names=["startnode","endnode", "rating", "timestamp"])

# fix timestamp
bitcoin_full.loc[:, "timestamp"] = pd.to_datetime(bitcoin_full.loc[:, "timestamp"], unit='s')

# Generate network using rating as edge label information
bitcoin_rating_ordering = ["startnode", "rating", "endnode"]
bitcoin_rating = bitcoin_full.loc[:, bitcoin_rating_ordering]

# Generate negative edges
#bitcoin_rating_labeled = negative_edge(bitcoin_rating, 'rating', limited=False, include_edge_label=True)
#bitcoin_rating_unlabeled = negative_edge(bitcoin_rating, 'rating', limited=False, include_edge_label=False)

# Generate test/training split and save as .txt
save_data(bitcoin_rating, 'BitcoinSign')

print("Bitcoin is ready..")

# ------------------------------------------
# FB15K-237 dataset
# ------------------------------------------

# TO DO

# ------------------------------------------
# WN18RR dataset
# ------------------------------------------

# TO DO