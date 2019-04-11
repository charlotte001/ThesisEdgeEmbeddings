# Importing modules
import pandas as pd
import numpy as np
import random
import os

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

def save_dataset(dataset, dataset_name, data_type, value):
    directory = "Datasets/" + dataset_name + "/" + value + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    dataset.to_csv(directory + data_type + ".txt", index=False, header=False, sep="\t")

def save_data(dataset, dataset_name, value):
    training, validation, test = data_split(dataset)
    save_dataset(training, dataset_name, 'train', value)
    save_dataset(validation, dataset_name, 'valid', value)
    save_dataset(test, dataset_name, 'test', value)

def rand_ent_except(entity, column, dataset):
    rand_index = random.randint(0, len(dataset) - 1)
    while(dataset.iat[rand_index, column] == entity):
        rand_index = random.randint(0, len(dataset) - 1)
    return dataset.iat[rand_index, column]

def generate_neg(dataset, neg_ratio):
    neg_batch = dataset
    for i in range(len(neg_batch)):
        if random.random() < 0.5:
            neg_batch.iat[i, 0] = rand_ent_except(neg_batch.iat[i, 0], 0, neg_batch) #flipping head
        else:
            neg_batch.iat[i, 2] = rand_ent_except(neg_batch.iat[i, 2], 2, neg_batch) #flipping tail
    return neg_batch

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
hepph_neg = generate_neg(dataset=hepph_full, neg_ratio=0.5)

# Generate test/training split and save as .txt
save_data(hepph_full, 'Cit-HepPh', value='Positive')
save_data(hepph_neg, 'Cit-HepPh', value='Negative')

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
bitcoin_rating_neg = generate_neg(dataset=bitcoin_rating, neg_ratio=0.5)

# Generate test/training split and save as .txt
save_data(bitcoin_rating, 'BitcoinSign', value="Positive")
save_data(bitcoin_rating_neg, 'BitcoinSign', value="Negative")

print("Bitcoin is ready..")

# ------------------------------------------
# FB15K-237 dataset
# ------------------------------------------

# load in dataset from datasets folder
FB15k_train = pd.read_csv("Datasets/FB15k-237/Original/train.txt", index_col=False, names=["startnode", "relation", "endnode"], sep="\t")
FB15k_valid = pd.read_csv("Datasets/FB15k-237/Original/valid.txt", index_col=False, names=["startnode", "relation", "endnode"], sep="\t")
FB15k_test = pd.read_csv("Datasets/FB15k-237/Original/test.txt", index_col=False, names=["startnode", "relation", "endnode"], sep="\t")

# generate negative edges
FB15k_train_neg = generate_neg(dataset=FB15k_train, neg_ratio=0.5)
FB15k_valid_neg = generate_neg(dataset=FB15k_valid, neg_ratio=0.5)
FB15k_test_neg = generate_neg(dataset=FB15k_test, neg_ratio=0.5)

# save datasets
save_dataset(FB15k_train, 'FB15K-237', data_type="train", value="Positive")
save_dataset(FB15k_valid, 'FB15K-237', data_type="valid", value="Positive")
save_dataset(FB15k_test, 'FB15K-237', data_type="test", value="Positive")
save_dataset(FB15k_train_neg, 'FB15K-237', data_type="train", value="Negative")
save_dataset(FB15k_valid_neg, 'FB15K-237', data_type="test", value="Negative")
save_dataset(FB15k_test_neg, 'FB15K-237', data_type="valid", value="Negative")

# ------------------------------------------
# WN18RR dataset
# ------------------------------------------

# load in dataset from datasets folder
WN18RR_train = pd.read_csv("Datasets/WN18RR/Original/train.txt", index_col=False, names=["startnode", "relation", "endnode"], sep="\t")
WN18RR_valid = pd.read_csv("Datasets/WN18RR/Original/valid.txt", index_col=False, names=["startnode", "relation", "endnode"], sep="\t")
WN18RR_test = pd.read_csv("Datasets/WN18RR/Original/test.txt", index_col=False, names=["startnode", "relation", "endnode"], sep="\t")

# generate negative edges
WN18RR_train_neg = generate_neg(dataset=WN18RR_train, neg_ratio=0.5)
WN18RR_valid_neg = generate_neg(dataset=WN18RR_valid, neg_ratio=0.5)
WN18RR_test_neg = generate_neg(dataset=WN18RR_test, neg_ratio=0.5)

# save datasets
save_dataset(WN18RR_train, 'WN18RR', data_type="train", value="Positive")
save_dataset(WN18RR_valid, 'WN18RR', data_type="valid", value="Positive")
save_dataset(WN18RR_test, 'WN18RR', data_type="test", value="Positive")
save_dataset(WN18RR_train_neg, 'WN18RR', data_type="train", value="Negative")
save_dataset(WN18RR_valid_neg, 'WN18RR', data_type="test", value="Negative")
save_dataset(WN18RR_test_neg, 'WN18RR', data_type="valid", value="Negative")