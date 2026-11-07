"""Train the 5 classifiers on the L&T Vehicle Loan Default dataset.

Reads the training split written by prepare_data.py (data/train_split.csv),
tunes each model with cross-validation on that split only, evaluates on
test_data.csv, writes the fitted pipelines to model/<name>.joblib and prints
the comparison table for the README.

Run from the repo root:  python model/train_models.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
TRAIN_PATH = ROOT / "data" / "train_split.csv"
TEST_PATH = ROOT / "test_data.csv"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"

TARGET = "loan_default"
RANDOM_STATE = 2202

# Hyperparameters are chosen by cross-validated ROC AUC, which ranks borrowers
# by risk independently of the 0.5 decision threshold.
TUNING_SCORE = "roc_auc"


def build_models():
    """(display name, file name, pipeline, parameter grid, CV folds) for each classifier.

    class_weight="balanced" is used wherever the classifier supports it, so the
    21.7% default class is not ignored. KNN and Naive Bayes have no such option.
    """
    return [
        (
            "Logistic Regression",
            "logistic_regression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]),
            {"clf__C": [0.01, 0.1, 1, 10]},
            5,
        ),
        (
            "Decision Tree",
            "decision_tree",
            # Trees split on thresholds, so no scaling is needed
            Pipeline([
                ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
            ]),
            {
                "clf__max_depth": [3, 4, 5, 6, 8, 10, 12],
                "clf__min_samples_leaf": [1, 20, 50, 100],
            },
            5,
        ),
        (
            "KNN",
            "knn",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]),
            {
                "clf__n_neighbors": [5, 11, 21, 31, 51],
                "clf__weights": ["uniform", "distance"],
            },
            5,
        ),
        (
            "Naive Bayes",
            "naive_bayes",
            # GaussianNB adds var_smoothing x (largest feature variance) to every
            # variance. Unscaled, the balance columns (variance ~1e11) would swamp
            # the 0/1 flags, so the features are standardised first.
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GaussianNB()),
            ]),
            {"clf__var_smoothing": np.logspace(-9, -1, 9)},
            5,
        ),
        (
            "Support Vector Machine",
            "svm",
            # No probability=True: it would refit with internal 5-fold CV and is
            # very slow. AUC uses decision_function instead.
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", class_weight="balanced", cache_size=1000)),
            ]),
            {"clf__C": [0.3, 1, 3]},
            3,  # fewer folds: each RBF fit on 8,000+ rows takes a while
        ),
    ]


def positive_scores(model, X):
    """Scores for the default class: probabilities if the model has them, else SVM decision values."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(model, X, y):
    """The 6 required metrics, with default (1) as the positive class."""
    y_pred = model.predict(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, positive_scores(model, X)),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def to_markdown(table):
    """Format the metrics table as Markdown, ready to paste into the README."""
    lines = [
        "| ML Model Name | " + " | ".join(table.columns) + " |",
        "|---" * (len(table.columns) + 1) + "|",
    ]
    for name, row in table.iterrows():
        lines.append(f"| {name} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
    return "\n".join(lines)


def main():
    features = json.loads(FEATURES_PATH.read_text())
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    X_train, y_train = train[features], train[TARGET]
    X_test, y_test = test[features], test[TARGET]

    print(f"scikit-learn {sklearn.__version__}  (pin this version in requirements.txt)")
    print(f"Train: {len(train):,} rows, Test: {len(test):,} rows, {len(features)} features")

    results = {}
    for name, filename, pipeline, grid, folds in build_models():
        start = time.perf_counter()
        search = GridSearchCV(
            pipeline,
            grid,
            scoring=TUNING_SCORE,
            cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        elapsed = time.perf_counter() - start

        joblib.dump(model, MODEL_DIR / f"{filename}.joblib")
        train_metrics = compute_metrics(model, X_train, y_train)
        test_metrics = compute_metrics(model, X_test, y_test)
        results[name] = test_metrics

        best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        print(f"\n{name}  ({elapsed:.1f}s)")
        print(f"  best params: {best}  (CV AUC {search.best_score_:.4f})")
        # Train vs test gap shows over-fitting, e.g. a deep decision tree
        print(
            f"  train: AUC {train_metrics['AUC']:.4f}, MCC {train_metrics['MCC']:.4f}"
            f"  |  test: AUC {test_metrics['AUC']:.4f}, MCC {test_metrics['MCC']:.4f}"
        )

    table = pd.DataFrame(results).T

    print("\nTest-set comparison (positive class = default):\n")
    print(to_markdown(table))
    print(f"\nSaved {len(results)} models to {MODEL_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
