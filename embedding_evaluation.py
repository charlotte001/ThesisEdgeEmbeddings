# Importing classifiers and evaluation approaches
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import auc, recall_score, precision_score, precision_recall_curve

# Importing pandas for datahandling
import pandas as pd

# ---------------------------------
# Functions
# ---------------------------------

def import_data(approach, dataset_name, include_edge_label=True):

    """ Function for importing the embeddings and the labels for the classification task"""

    if include_edge_label == True:
        data_X_training = 1
        data_X_test = 1
        data_y_training = pd.read_csv("Datasets/"+ dataset_name +"/edge_y_training.txt")
        data_y_test = pd.read_csv("Datasets/"+ dataset_name +"/edge_y_training.txt")
    else:
        data_X_training = 1
        data_X_test = 1
        data_y_training = pd.read_csv("Datasets/"+ dataset_name +"/noedge_y_training.txt")
        data_y_test = pd.read_csv("Datasets/"+ dataset_name +"/noedge_y_training.txt")
    
    return(data_X_training, data_X_test, data_y_training, data_y_test)

# ---------------------------------
# Import of embeddings and labels
# ---------------------------------

X_training, X_test, y_training, y_test = import_data('SimplE', 'BitcoinSign', include_edge_label=True)

# ---------------------------------
# Gradient Boosting
# ---------------------------------

# Fit Gradient Boosting

gbclass = GradientBoostingClassifier()
gbclass = gbclass.fit(X_training, y_training)

# Predict Gradient Boosting 

gb_predictions = gbclass.predict(X_test)

# Score Gradient Boosting

# TO DO

# ---------------------------------
# Logistic Regression
# ---------------------------------

# Fit Logistic Regression

logreg = LogisticRegression()
logreg = logreg.fit(X_training, y_training)

# Predict Logistic Regression

lr_predictions = logreg.predict(X_test)

# Score Logistic Regression

# TO DO