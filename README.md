# Dead Internet Theory – Bot Detection Project

This project investigates whether Reddit activity can be used to detect automated bot accounts. Using a dataset of Reddit comments and metadata, we build and compare several machine learning models to classify whether a comment was made by a human or a bot.

## Team
- Alda Zeneli – Logistic Regression
- Luke Cargill – Decision Tree
- Tahjae Jackson – Random Forest
- Hien Bui – K‑Nearest Neighbors
- Olivia McField - Evaluation 

## Project Goal
The goal of this project is to explore how different machine learning models perform on the task of bot detection. We preprocess Reddit comment metadata, train multiple classification models, and compare their performance.

## Dataset
The dataset contains Reddit comment information with features such as:

- account_age_days
- user_karma
- reply_delay_seconds
- sentiment_score
- avg_word_length
- contains_links
- subreddit

The target variable is:

- **is_bot_flag**
  - `0` = human
  - `1` = bot

## Data Processing
The `cleaning.py` script performs preprocessing steps including:

- Removing identifier columns
- Converting boolean variables to numeric values
- Encoding categorical variables (subreddit)
- Handling missing values
- Splitting the dataset into training and testing sets
- Feature scaling for models that require normalization

After running `cleaning.py`, the following datasets are generated:

```
data/X_train_clean.csv
- Training features used to train models

data/X_test_clean.csv
- Testing features used to evaluate models

data/y_train.csv
- Training labels (bot vs human)

data/y_test.csv
- Testing labels
```

## Using the Cleaned Data

The cleaned datasets are used by all models in the project.

```
X_train_clean.csv – training features (input variables used to train the models)

X_test_clean.csv – testing features (input variables used to evaluate the models)

y_train.csv – training labels (whether each training example is a bot or human)

y_test.csv – testing labels (true bot/human labels used to measure model performance)
```

Example code to load the datasets before training a model:

```python
import pandas as pd

X_train = pd.read_csv("data/X_train_clean.csv")
X_test = pd.read_csv("data/X_test_clean.csv")

y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()
```

All models (Logistic Regression, KNN, Decision Tree, and Random Forest) use these same datasets to ensure a fair comparison.

## Models Implemented
The following models are implemented and compared:

- Logistic Regression
- K‑Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

Each model is trained using the same cleaned dataset to ensure a fair comparison.

## Project Workflow

```
Raw Dataset
    ↓
cleaning.py
    ↓
Train/Test Split
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Performance Comparison
```

## Running the Project

1. Install dependencies:

```
pip install pandas scikit-learn
```

2. Run the cleaning pipeline:

```
python cleaning.py
```

3. Train and evaluate the models using the cleaned datasets.

## Evaluation
Model performance is evaluated using common classification metrics such as:

- Accuracy
- Precision
- Recall
- ROC‑AUC

These metrics allow us to compare how well each model detects bot activity.

## Repository Structure

```
Project/
│
├── cleaning.py
├── README.md
├── data/
│   ├── X_train_clean.csv
│   ├── X_test_clean.csv
│   ├── y_train.csv
│   └── y_test.csv
│
└── models/
    ├── logistic_regression.py
    ├── knn.py
    ├── decision_tree.py
    └── random_forest.py
|
|
|___ evaluation.py - used to evaluate the model performation based on various evalutaion metrics
```

## Course
ENGS 106 – Principles of Machine Learning
Dartmouth College