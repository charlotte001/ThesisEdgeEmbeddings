# Importing modules
import pandas as pd

print("Modules loaded, starting with datasets..")

# ------------------------------------------
# HepPh dataset
# ------------------------------------------

# Load in datasets from datasets folder
hepph_net = pd.read_csv("Datasets/cit-HepPH/original/Cit-HepPh.txt", sep="\t", skiprows=4, index_col=False, names=["startnode","endnode"])
hepph_time = pd.read_csv("Datasets/cit-HepPH/original/cit-HepPh-dates.txt", sep="\t", skiprows=1, index_col=False, names=["startnode","timestamp"])

# Merge datasets on startnode column
hepph_merged = hepph_net.merge(hepph_time, how="left" ,on="startnode").dropna()

# Create test/training split using a 30/70 fraction
hepph_merged_test = hepph_merged.sample(frac=0.3)

# Using the index of the selection of the training set to drop it from 
hepph_merged_training = hepph_merged.drop(hepph_merged_test.index)

# Output the training and test set to txt file with edge labels
hepph_merged_training.to_csv("Datasets/cit-HepPH/Cit-HepPh-training-labeled.txt", index=False)
hepph_merged_test.to_csv("Datasets/cit-HepPH/Cit-HepPh-test-labeled.txt", index=False)

# Output the training and test set to txt file without edge labels
hepph_merged_training_unlabeled = hepph_merged_training.loc[:, ["startnode", "endnode"]]
hepph_merged_training_unlabeled.to_csv("Datasets/cit-HepPH/Cit-HepPh-training-unlabeled.txt", index=False)

hepph_merged_test_unlabeled = hepph_merged_test.loc[:, ["startnode", "endnode"]]
hepph_merged_test_unlabeled.to_csv("Datasets/cit-HepPH/Cit-HepPh-test-unlabeled.txt", index=False)

print("HepPH is ready..")

# ------------------------------------------
# Bitcoin dataset
# ------------------------------------------

# load in dataset from datasets folder
bitcoin_full = pd.read_csv("Datasets/BitcoinSign/original/soc-sign-bitcoinotc.csv", index_col=False, names=["startnode","endnode", "rating", "timestamp"])

# fix timestamp
bitcoin_full.loc[:, "timestamp"] = pd.to_datetime(bitcoin_full.loc[:, "timestamp"], unit='s')

# Generate network using rating as edge label information
bitcoin_rating_ordering = ["startnode", "rating", "endnode"]
bitcoin_rating = bitcoin_full.loc[:, bitcoin_rating_ordering]

# Create test/training split using a 30/70 fraction
bitcoin_rating_test = bitcoin_rating.sample(frac=0.3)

# Using the index of the selection of the training set to drop it from main dataframe
bitcoin_rating_training = bitcoin_rating.drop(bitcoin_rating_test.index)

# Output the training and test set to txt file with edge labels
bitcoin_rating_training.to_csv("Datasets/BitcoinSign/BitcoinSign-training-labeled.txt", index=False)
bitcoin_rating_test.to_csv("Datasets/BitcoinSign/BitcoinSign-test-labeled.txt", index=False)

# Output the training and test set to txt file without edge labels
bitcoin_rating_training_unlabeled = bitcoin_rating_training.loc[:, ["startnode", "endnode"]]
bitcoin_rating_training_unlabeled.to_csv("Datasets/BitcoinSign/BitcoinSign-training-unlabeled.txt", index=False)

bitcoin_rating_test_unlabeled = bitcoin_rating_test.loc[:, ["startnode", "endnode"]]
bitcoin_rating_test_unlabeled.to_csv("Datasets/BitcoinSign/BitcoinSign-test-unlabeled.txt", index=False)

print("Bitcoin is ready..")

# ------------------------------------------
# FB15K-237 dataset
# ------------------------------------------

# TO DO

# ------------------------------------------
# WN18RR dataset
# ------------------------------------------

# TO DO