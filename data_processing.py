# Importing modules
import pandas as pd
import random

print("Modules loaded, starting with datasets..")

#-------------------------------------------
# Create functions for processing
#-------------------------------------------

def negative_edge(dataset, edge_label, limited=True, include_edge_label=True):
    
    """ Function to generate negative or fake edges """
    """ The function accepts a column name for the edge label"""
    """ By setting limited to False, the function is run on the length of the entire dataset """
    """ By setting include_edge_label to False, no edge label is included in the final output """
    
    # Setting variables and dicts used in the function
    negative_edge_dict = {}
    dataset = dataset.reset_index(drop=True)
    data_len = len(dataset) - 1
    
    # Snippet to limit generation of edges for testing purposes
    if limited == False:
        range_limit = data_len
    else:
        range_limit = 20
        
    print("Start negative edge generation with label = " + str(include_edge_label))
    
    # Generating negative edges by selecting from the original dataset using random indexes 
    for i in range(0, range_limit):
        startindex = random.randint(0, data_len)
        endindex = random.randint(0, data_len)
        timeindex = random.randint(0, data_len)
        
        negative_edge_dict[i + data_len]= {'startnode': dataset.iat[startindex, 0],
                                edge_label: dataset.iat[timeindex, 1],
                                'endnode': dataset.iat[endindex, 2],
                                }
            
    print("Edge generation is completed, now transforming and appending..")
    
    # Transforming dictionary into dataframe and appending it to original dataset
    negative_edge_df = pd.DataFrame.from_dict(negative_edge_dict, orient='index')

    print("Nodes created: " + str(negative_edge_df['startnode'].nunique()) + ", edges created: " + str(negative_edge_df['startnode'].count()))
    dataset = dataset.append(negative_edge_df)
        
    # If-statement to determine output format (include/not-include edge label)
    if include_edge_label == True:
        return(dataset.loc[:, ["startnode", edge_label, "endnode"]])
    else:
        return(dataset.loc[:, ["startnode","endnode"]])

    print("Negative edge creation procedure completed for this iteration.")

def create_test_split(dataset, dataset_name, labeled=True):
    
    """ Function that creates a 30/70 test/training split """
    """ Function takes dataset_name to name the .txt files """
    """ Includes labeled in the filename if labeled=True"""
    
    # Create test/training split using a 30/70 fraction
    dataset_test = dataset.sample(frac=0.3)

    # Using the index of the selection of the training set to drop it from original dataset 
    dataset_training = dataset.drop(dataset_test.index)

    if labeled == True:
        # Output the training and test set to txt file with edge labels
        dataset_training.to_csv("Datasets/"+ dataset_name +"/"+ dataset_name +"-training-labeled.txt", index=False)
        dataset_test.to_csv("Datasets/"+ dataset_name +"/"+ dataset_name +"-test-labeled.txt", index=False)
    else:
        # Output the training and test set to txt file without edge labels
        dataset_training.to_csv("Datasets/"+ dataset_name +"/"+ dataset_name +"-training-unlabeled.txt", index=False)
        dataset_test.to_csv("Datasets/"+ dataset_name +"/"+ dataset_name +"-test-unlabeled.txt", index=False)

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
hepph_labeled = negative_edge(hepph_full, 'timestamp', limited=False, include_edge_label=True)
hepph_unlabeled = negative_edge(hepph_full, 'timestamp', limited=False, include_edge_label=False)

# Generate test/training split and save as .txt
create_test_split(hepph_labeled, 'Cit-HepPh', labeled=True)
create_test_split(hepph_unlabeled, 'Cit-HepPh', labeled=False)

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
bitcoin_rating_labeled = negative_edge(bitcoin_rating, 'rating', limited=False, include_edge_label=True)
bitcoin_rating_unlabeled = negative_edge(bitcoin_rating, 'rating', limited=False, include_edge_label=False)

# Generate test/training split and save as .txt
create_test_split(bitcoin_rating_labeled, 'BitcoinSign', labeled=True)
create_test_split(bitcoin_rating_labeled, 'BitcoinSign', labeled=False)

print("Bitcoin is ready..")

# ------------------------------------------
# FB15K-237 dataset
# ------------------------------------------

# TO DO

# ------------------------------------------
# WN18RR dataset
# ------------------------------------------

# TO DO