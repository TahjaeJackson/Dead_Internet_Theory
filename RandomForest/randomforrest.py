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

for key, value in results.items():
    if key not in ["y_prob", "y_true"]:
        print(f"{key}: {value}")

metrics_df = pd.DataFrame([results])
metrics_df.to_csv("random_forest_metrics.csv", index=False)


# Verifying that the model perfomance is not due to training and testing dataset leakage

print("\nDataset Shapes")
print("Train:", X_train_df.shape)
print("Test:", X_test_df.shape)

overlap = pd.merge(X_train_df, X_test_df)
print("Overlapping rows:", len(overlap))
print(X_train_df.columns)

temp = X_train_df.copy()
temp["target"] = y_train

print(temp.corr()["target"].sort_values(ascending=False))

# cross-validation test 
scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="roc_auc")
print("\nCross Validation ROC-AUC Scores:", scores)
print("Mean ROC-AUC:", scores.mean())

# identifying the most important features 

importances = rf_model.feature_importances_
features = X_train_df.columns

importance_df = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values("importance", ascending=True)

print(importance_df)

green = "#2E8B57"

fig, ax = plt.subplots(figsize=(10,6))

ax.barh(
    importance_df["feature"],
    importance_df["importance"],
    color=green
)

ax.set_xlabel("Feature Importance", fontsize=12)
ax.set_ylabel("Feature", fontsize=12)
ax.set_title("Random Forest Feature Importance", fontsize=14)

ax.grid(axis="x", linestyle="--", alpha=0.6)

# green border for poster aesthetic
for spine in ax.spines.values():
    spine.set_edgecolor(green)
    spine.set_linewidth(2)

plt.tight_layout()

# save high resolution image for poster
plt.savefig("random_forest_feature_importance.png", dpi=600, bbox_inches="tight", facecolor="white")

plt.close()


# Predict bots and compute bot percentage per subreddit

# Predict bots and compute bot percentage per subreddit

# Predict probabilities
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Convert to predicted labels
y_pred = (y_prob >= 0.5).astype(int)

# Attach predictions back to dataframe
X_test_analysis = X_test_df.copy()
X_test_analysis["predicted_bot"] = y_pred
X_test_analysis["bot_probability"] = y_prob


# reconstruct subreddit label from one-hot columns
subreddit_cols = [
    "subreddit_gaming",
    "subreddit_pics",
    "subreddit_politics",
    "subreddit_technology",
    "subreddit_worldnews"
]

X_test_analysis["subreddit"] = (
    X_test_analysis[subreddit_cols]
    .idxmax(axis=1)
    .str.replace("subreddit_", "")
)

# compute bot percentage per subreddit
bot_df = (
    X_test_analysis
    .groupby("subreddit")["bot_probability"]
    .mean()
    .reset_index()
)

bot_df["Bot Percentage"] = bot_df["bot_probability"] * 100
bot_df = bot_df.drop(columns="bot_probability")

bot_df = bot_df.sort_values("Bot Percentage", ascending=False)

print("\nPredicted Bot Percentage by Subreddit")
print(bot_df)

# Plot bar graph of bot percentage by subreddit

# Plot bar graph of bot percentage by subreddit

green = "#2E8B57"   # main poster green

fig, ax = plt.subplots(figsize=(8,5))

# create varying green colors
colors = plt.cm.Greens(
    np.linspace(0.4, 0.9, len(bot_df))
)

ax.bar(
    bot_df["subreddit"],
    bot_df["Bot Percentage"],
    color=colors
)

# axis labels
ax.set_ylabel("Predicted Bot Percentage (%)", fontsize=12)
ax.set_xlabel("Subreddit", fontsize=12,)

# title
ax.set_title(
    "Predicted Bot Activity by Subreddit",
    fontsize=14
)

# tick colors
ax.tick_params(axis="x")
ax.tick_params(axis="y")

# rotate labels
ax.set_xticklabels(bot_df["subreddit"], rotation=30)

# grid
ax.grid(axis="y", linestyle="--", alpha=0.6)

# green border
for spine in ax.spines.values():
    spine.set_edgecolor(green)
    spine.set_linewidth(2)

plt.tight_layout()

# save poster-quality image
plt.savefig(
    "bot_percentage_by_subreddit.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()