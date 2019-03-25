# Importing classifiers and evaluation approaches
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import auc, recall_score, precision_score, precision_recall_curve

# Importing pandas for datahandling
import pandas as pd

# create pipeline for classfication task and evaluation task