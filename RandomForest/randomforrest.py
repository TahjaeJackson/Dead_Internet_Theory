"""
Author: Tahjae Jackson 
Description: Create and implement random forest model for the prediction of bots for the dead internet theory
Date: March 5, 2026

"""

# impoorting the necessary libraries
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import pandas as pd
import sys
import os
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# ensuring that the path to the evaluation module can be accessed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation import evaluate_model

# Loading datasets
X_train_df = pd.read_csv("DATA/X_train_clean.csv")
X_test_df = pd.read_csv("DATA/X_test_clean.csv")
y_train_df = pd.read_csv("DATA/y_train.csv")
y_test_df = pd.read_csv("DATA/y_test.csv")

# convert X to arrays
# keep y as pandas
y_train = y_train_df.squeeze()
y_test = y_test_df.squeeze()
X_train = X_train_df.values
X_test = X_test_df.values


# training the random forrest model 
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# evaluating the model using the evaluation code that is imported 
results = evaluate_model(
    rf_model,
    X_test,
    y_test,
    threshold=0.5,
    plot_prefix="random_forest"
)


# displaying metrics to be used in presentation 

print("\nRandom Forest Performance")
print("--------------------------")

for key, value in results.items():
    if key not in ["y_prob", "y_true"]:
        print(f"{key}: {value}")

metrics_df = pd.DataFrame([results])
metrics_df.to_csv("random_forest_metrics.csv", index=False)


# Verifying that the model perfomance is not due to training and testing dataset leakage

print(X_train_df.shape)
print(X_test_df.shape)

overlap = pd.merge(X_train_df, X_test_df)
print(len(overlap))
print(X_train_df.columns)

temp = X_train_df.copy()
temp["target"] = y_train

print(temp.corr()["target"].sort_values(ascending=False))

# cross-validation test 
scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="roc_auc")
print(scores)
print(scores.mean())


# Identifying the inmportant features 

importances = rf_model.feature_importances_
features = X_train_df.columns

importance_df = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values("importance", ascending=False)

print(importance_df)

plt.figure()
plt.barh(importance_df["feature"], importance_df["importance"])
plt.gca().invert_yaxis()
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("random_forest_feature_importance.png")