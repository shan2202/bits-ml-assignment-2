"""Train the 5 classifiers on the L&T Vehicle Loan Default dataset.

Reads data/train.csv (from Kaggle, not committed), writes the fitted
pipelines to model/*.joblib and the held-out split to test_data.csv.
"""

# TODO: load data/train.csv
# TODO: preprocessing (drop IDs/constants, age at disbursal, durations -> months, encode categoricals)
# TODO: stratified subsample + stratified 80/20 split; save test split to test_data.csv
# TODO: train Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM as Pipelines
# TODO: compute Accuracy, AUC, Precision, Recall, F1, MCC; print the comparison table
# TODO: joblib.dump each pipeline into model/
