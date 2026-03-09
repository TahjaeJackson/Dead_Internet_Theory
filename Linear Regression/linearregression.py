import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report


import matplotlib.pyplot as plt

# use Helvetica for cleaner poster-style plots
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 11


# load cleaned datasets
X_train = pd.read_csv("data/X_train_clean.csv")
X_test = pd.read_csv("data/X_test_clean.csv")

y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()


# initialize logistic regression model
model = LogisticRegression(max_iter=1000)


# train model
model.fit(X_train, y_train)


# predictions
y_pred = model.predict(X_test)

# probability predictions for ROC-AUC
y_prob = model.predict_proba(X_test)[:, 1]


# evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)


print("Logistic Regression Results")
print("---------------------------")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

coefficients = pd.Series(model.coef_[0], index=X_train.columns)
print("\nFeature Importance:")
print(coefficients.sort_values(ascending=False))

# create feature importance plot styled for the poster
coeff_sorted = coefficients.sort_values(ascending=True)

plt.figure(figsize=(10,6))

# Dartmouth-style green bars
coeff_sorted.plot(kind='barh', color='#0B6B3A')

# vertical reference line
plt.axvline(0, color='black', linewidth=1)

plt.title("Logistic Regression Feature Importance", fontsize=14)
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")

plt.tight_layout()

# save high quality image for poster
plt.savefig("logistic_regression_feature_importance.png", dpi=300)

plt.show()


print(X_train.columns)


# with open("logistic_regression_results.md", "w") as f:
#     f.write("# Logistic Regression Results\n\n")
#     f.write("## Model Performance\n")
#     f.write(f"- Accuracy: {accuracy:.4f}\n")
#     f.write(f"- Precision: {precision:.4f}\n")
#     f.write(f"- Recall: {recall:.4f}\n")
#     f.write(f"- ROC-AUC: {roc_auc:.4f}\n\n")

#     f.write("## Dataset Sizes\n")
#     f.write(f"- Training samples: {X_train.shape[0]}\n")
#     f.write(f"- Test samples: {X_test.shape[0]}\n")
#     f.write(f"- Number of features: {X_train.shape[1]}\n\n")

#     f.write("## Classification Report\n")
#     f.write("```\n")
#     f.write(classification_report(y_test, y_pred))
#     f.write("\n```\n\n")

#     f.write("## Feature Importance\n")
#     coeff_sorted = coefficients.sort_values(ascending=False)
#     for feature, value in coeff_sorted.items():
#         f.write(f"- {feature}: {value:.4f}\n")