# Vehicle Loan Default Prediction

## a. Problem statement

Vehicle loans in India are often given to borrowers with little or no credit history, and a loan that defaults on its very first EMI is a direct loss to the lender. The goal of this project is to predict, at the time of disbursal, whether a borrower will **default on the first EMI** of a vehicle loan, using the loan details, the borrower's KYC/demographic information and their credit bureau history.

This is a **binary classification** problem (`loan_default`: 1 = defaulted, 0 = paid). A useful model should catch as many likely defaulters as possible without rejecting too many good borrowers, so models are compared not only on accuracy but on AUC, precision, recall, F1 and MCC, which are more informative when the classes are imbalanced.

Five classifiers are trained and compared: Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and a Support Vector Machine. The trained models are served through a Streamlit app that evaluates them on uploaded test data.

## b. Dataset description

**Source:** [L&T Vehicle Loan Default Prediction](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction) on Kaggle (uploaded by Mamta Dhaker; data from L&T Financial Services, India). Only `train.csv` is used, since it is the only file with labels.

| Property | Value |
|---|---|
| Rows | 233,154 loans |
| Columns | 41 (39 input columns + `UniqueID` + target) |
| Target | `loan_default`: 1 if the borrower defaulted on the first EMI, else 0 |
| Class balance | 182,543 non-default (78.29%) vs 50,611 default (**21.71%**) |
| Missing values | Only `Employment.Type` (7,661 rows, 3.3%) |

**Feature groups in the raw data**

| Group | Columns |
|---|---|
| Loan | `disbursed_amount`, `asset_cost`, `ltv` (loan-to-value %), `DisbursalDate` |
| Borrower | `Date.of.Birth`, `Employment.Type`, ID-proof flags (`Aadhar_flag`, `PAN_flag`, `VoterID_flag`, `Driving_flag`, `Passport_flag`, `MobileNo_Avl_Flag`) |
| Credit bureau | **Score:** `PERFORM_CNS.SCORE`, `PERFORM_CNS.SCORE.DESCRIPTION`<br>**Primary accounts:** `PRI.NO.OF.ACCTS`, `PRI.ACTIVE.ACCTS`, `PRI.OVERDUE.ACCTS`, `PRI.CURRENT.BALANCE`, `PRI.SANCTIONED.AMOUNT`, `PRI.DISBURSED.AMOUNT`, `PRIMARY.INSTAL.AMT`<br>**Secondary accounts:** `SEC.NO.OF.ACCTS`, `SEC.ACTIVE.ACCTS`, `SEC.OVERDUE.ACCTS`, `SEC.CURRENT.BALANCE`, `SEC.SANCTIONED.AMOUNT`, `SEC.DISBURSED.AMOUNT`, `SEC.INSTAL.AMT`<br>**Last 6 months:** `NEW.ACCTS.IN.LAST.SIX.MONTHS`, `DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS`<br>**History:** `AVERAGE.ACCT.AGE`, `CREDIT.HISTORY.LENGTH`, `NO.OF_INQUIRIES` |
| Identifiers | `UniqueID`, `branch_id`, `supplier_id`, `manufacturer_id`, `Current_pincode_ID`, `State_ID`, `Employee_code_ID` |

**Observations from EDA**

- The data is imbalanced: a model that always predicts "no default" would already score about 78% accuracy, so accuracy alone is misleading here.
- Loan amounts are heavily right-skewed: the median `disbursed_amount` is ₹53,803 but the maximum is ₹9,90,572; `asset_cost` behaves the same way.
- About half of the borrowers (116,950, 50.2%) have "No Bureau History Available" and a bureau score of 0, i.e. they are new to credit.

**Preprocessing** (`model/prepare_data.py`)

1. Dropped the 7 identifier columns: they are high-cardinality codes with no meaning outside this lender.
2. Replaced `Date.of.Birth` and `DisbursalDate` with **age at disbursal** in years. The dates use 2-digit years, so birth dates that parsed into the future were moved back 100 years (resulting ages: 18.0 to 69.1, median 32.7).
3. Converted `AVERAGE.ACCT.AGE` and `CREDIT.HISTORY.LENGTH` from text like `"1yrs 11mon"` into a number of months.
4. Merged each primary/secondary bureau pair into one total (e.g. `TOTAL.CURRENT.BALANCE`), since secondary accounts are rare and the pairs are strongly correlated.
5. Added an `ANY.DELINQUENCY` flag: 1 if the borrower has any delinquent account in the last 6 months or any overdue account.
6. Filled missing `Employment.Type` with `"Missing"` and one-hot encoded it (3 columns).
7. Grouped the 20 values of `PERFORM_CNS.SCORE.DESCRIPTION` (13 risk grades A–M, 6 "Not Scored" reasons and "No Bureau History Available") into 7 risk bands and one-hot encoded them (see the table below).
8. Dropped `MobileNo_Avl_Flag`, which is 1 for every row.

| Risk band | Original descriptions | Rows (full data) |
|---|---|---|
| No History | No Bureau History Available | 116,950 |
| Not Scored | 6 "Not Scored: …" reasons (e.g. sufficient history not available, inactive, only a guarantor) | 12,835 |
| Very Low Risk | A, B, C, D | 50,728 |
| Low Risk | E, F, G | 18,294 |
| Medium Risk | H, I | 12,412 |
| High Risk | J, K | 12,025 |
| Very High Risk | L, M | 9,910 |

This leaves **33 numeric features**. All of these steps work row by row with fixed category lists, so nothing is learned from the data before the split. Feature scaling is done inside each model's pipeline, fitted on the training split only.

**Subsampling and split**

- A **stratified subsample of 15,000 rows** (`random_state=2202`) was taken from the cleaned data, to keep SVM training time and the KNN model size practical while preserving the 21.7% default rate.
- The subsample was split **80/20, stratified**: 12,000 rows for training and 3,000 rows for testing. Both keep the 21.7% default rate.
- The 3,000-row test split is saved as `test_data.csv` (33 features + `loan_default`). This is the file to upload in the Streamlit app.

## c. GitHub repository link

https://github.com/shan2202/bits-ml-assignment-2

**Live Streamlit app:** TODO

## d. Models used

All five models are scikit-learn pipelines trained on the 12,000-row training split (`model/train_models.py`). Hyperparameters were chosen by stratified cross-validation on the training split only, using ROC AUC; `class_weight="balanced"` was used wherever the classifier supports it.

| ML Model Name | Pipeline | Selected hyperparameters |
|---|---|---|
| Logistic Regression | StandardScaler + LogisticRegression (balanced) | C = 0.1 |
| Decision Tree | DecisionTreeClassifier (balanced) | max_depth = 4, min_samples_leaf = 50 |
| KNN | StandardScaler + KNeighborsClassifier | n_neighbors = 151, weights = distance |
| Naive Bayes | StandardScaler + GaussianNB | var_smoothing = 1.0 |
| Support Vector Machine | StandardScaler + SVC, RBF kernel (balanced) | C = 0.3 |

### Comparison table

Evaluated on the held-out `test_data.csv` (3,000 rows, 21.7% default). Positive class = default (1). AUC uses predicted probabilities, or the decision function for the SVM.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.5730 | 0.6251 | 0.2816 | 0.6237 | 0.3880 | 0.1507 |
| Decision Tree | 0.4920 | 0.5987 | 0.2598 | 0.7250 | 0.3825 | 0.1286 |
| KNN | 0.7827 | 0.6008 | 0.0000 | 0.0000 | 0.0000 | -0.0096 |
| Naive Bayes | 0.6807 | 0.5964 | 0.2871 | 0.3180 | 0.3017 | 0.0957 |
| Support Vector Machine | 0.5673 | 0.6192 | 0.2774 | 0.6190 | 0.3831 | 0.1419 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| Overall, winner for this dataset? | |

## Streamlit app

The app (`app.py`) loads the five saved pipelines and evaluates them on an uploaded CSV; it never retrains.

- **Upload** a test CSV with the 33 feature columns and `loan_default` (the sidebar has a button to download `test_data.csv`). The file is checked for missing columns, empty or non-numeric values and a valid 0/1 target.
- **Choose a model** from the dropdown.
- **Selected-model tab:** the six metrics, the confusion matrix with a plain-language summary, the classification report and the ROC curve.
- **Compare all models tab:** all six metrics for the five models, ROC curves on one chart and MCC per model.
- **Predictions tab:** actual vs predicted label and score for every row, downloadable as CSV.

Run it locally with `streamlit run app.py`.

## Repository structure

```
├── app.py                  # Streamlit app
├── requirements.txt
├── README.md
├── test_data.csv           # held-out 20% test split (generated by model/prepare_data.py)
├── .streamlit/config.toml  # app theme
├── model/
│   ├── __init__.py
│   ├── prepare_data.py     # cleaning, feature engineering, subsample + split
│   ├── train_models.py     # training + metrics
│   ├── feature_columns.json
│   └── *.joblib            # saved pipelines for the 5 classifiers
└── data/                   # place Kaggle train.csv here (git-ignored)
```
