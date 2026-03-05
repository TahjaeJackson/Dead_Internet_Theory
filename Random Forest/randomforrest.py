"""
Author: Tahjae Jackson 
Description: Create and implement random forest model for the prediction of bots for the dead internet theory
Date: March 5, 2026

"""

# impoorting the necessary libraries
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import pandas as pd

# getting test and train dataset
X_train_df = pd.read_csv("../DATA/X_train_clean.csv")
X_test_df = pd.read_csv("../DATA/X_test_clean.csv")
y_train_df = pd.read_csv("../DATA/y_train.csv")
y_test_df = pd.read_csv("../DATA/y_test.csv")



print(X_train_df)
