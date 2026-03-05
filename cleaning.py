import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/reddit_dead_internet_analysis_2026.csv")

# Drop identifiers
df = df.drop(columns=["comment_id", "bot_type_label"])

# Convert boolean columns
df["contains_links"] = df["contains_links"].astype(int)
df["is_bot_flag"] = df["is_bot_flag"].astype(int)

# One-hot encode subreddit
df = pd.get_dummies(df, columns=["subreddit"])

# Handle missing values
df = df.dropna()

# Split features and label (drop bot_probability to avoid data leakage if present)
drop_cols = ["is_bot_flag"]
if "bot_probability" in df.columns:
    drop_cols.append("bot_probability")

X = df.drop(columns=drop_cols)
y = df["is_bot_flag"]

# Save feature names for later model interpretation
feature_names = X.columns

# Train test split (use stratify to preserve bot/human ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (fit only on training data to avoid leakage)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Optional: save cleaned datasets for teammates
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

X_train_df.to_csv("data/X_train_clean.csv", index=False)
X_test_df.to_csv("data/X_test_clean.csv", index=False)

y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)