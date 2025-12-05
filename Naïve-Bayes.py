import pandas as pd
from sklearn.model_selection import train_test_split
import math

# Define categorical columns
categoral_columns = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
    "income"
]

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Keep only required categorical columns
    df = df[categoral_columns]

    # Treat "?" as its own category
    df.replace("?", "Missing", inplace=True)

    # Encode categories as integers
    for col in categoral_columns:
        df[col] = df[col].astype("category").cat.codes

    # Split X and y
    X = df.drop("income", axis=1)
    y = df["income"]

    # 70% training, 30% validation + test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Split 15% validation, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# Analyze class distributions and feature-target relationships
def analyze_data(X, y):
    print("Class distribution:")
    print(y.value_counts(normalize=True))

    for col in X.columns:
        print(f"\nFeature: {col}")
        feature_target_dist = pd.crosstab(X[col], y, normalize='index')
        print(feature_target_dist)

# Train Naive Bayes model
def train_naive_bayes(X_train, y_train, alpha=1):
    class_priors = {}
    feature_likelihoods = {}

    classes = y_train.unique()
    num_classes = len(classes)
    total_samples = len(y_train)

    # CLASS PRIORS
    for cls in classes:
        count_cls = (y_train == cls).sum()
        class_priors[cls] = (count_cls + alpha) / (total_samples + alpha * num_classes)

    # FEATURE LIKELIHOODS
        # Get all samples of class cls
        X_cls = X_train[y_train == cls]
        feature_likelihoods[cls] = {}

        for col in X_train.columns:
            # Get unique values of the feature
            feature_values = X_train[col].unique()
            total_count = len(X_cls)
            num_feature_values = len(feature_values)

            likelihoods = {}

            for feature_value in feature_values:
                # Count of feature_value in class cls
                count_value = (X_cls[col] == feature_value).sum()

                likelihoods[feature_value] = (count_value + alpha) / (
                    total_count + alpha * num_feature_values
                )

            feature_likelihoods[cls][col] = likelihoods
    return class_priors, feature_likelihoods

def predict_single_log(x, class_priors, feature_likelihoods):
    best_class = None
    best_log_prob = -float("inf")

    for cls in class_priors:
        # Start with log prior
        log_prob = math.log(class_priors[cls])

        # Add log likelihoods
        for col, _ in x.items():
            likelihood = feature_likelihoods[cls][col].get(x[col], 1e-8)  # Smoothing for unseen feature values
            log_prob += math.log(likelihood)

        # Choose class with max log probability
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_class = cls

    return best_class

def predict_log(X, class_priors, feature_likelihoods):
    preds = []
    for _, row in X.iterrows():
        preds.append(predict_single_log(row, class_priors, feature_likelihoods))
    return preds

