"""Prepare the L&T Vehicle Loan Default data for training.

Reads the raw Kaggle file data/train.csv and writes:
  - data/train_split.csv        80% split used to train the models (git-ignored)
  - test_data.csv               20% held-out split: cleaned features + loan_default
  - model/feature_columns.json  feature names, in the order the models expect

Every transformation here is row-by-row with fixed category lists, so nothing
is learned from the data and the train/test split cannot leak. Scaling is left
to the model pipelines in train_models.py.

Run from the repo root:  python model/prepare_data.py
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "train.csv"
TRAIN_OUT = ROOT / "data" / "train_split.csv"
TEST_OUT = ROOT / "test_data.csv"
FEATURES_OUT = ROOT / "model" / "feature_columns.json"

TARGET = "loan_default"
RANDOM_STATE = 2202
SUBSAMPLE_SIZE = 15000
TEST_SIZE = 0.2

# Row IDs and branch/supplier/dealer/staff/location codes: high-cardinality
# labels with no meaning outside this lender, so they are dropped.
ID_COLUMNS = [
    "UniqueID",
    "branch_id",
    "supplier_id",
    "manufacturer_id",
    "Current_pincode_ID",
    "State_ID",
    "Employee_code_ID",
]

EMPLOYMENT_TYPES = ["Salaried", "Self employed", "Missing"]

# PERFORM_CNS.SCORE.DESCRIPTION has ~20 values (A-M risk grades plus several
# "Not Scored: ..." reasons); they are grouped into these bands.
CNS_BANDS = [
    "No History",
    "Not Scored",
    "Very Low Risk",
    "Low Risk",
    "Medium Risk",
    "High Risk",
    "Very High Risk",
]

# Primary and secondary bureau accounts are merged into one total each.
# Secondary accounts are rare, and the pairs are highly correlated.
PRI_SEC_TOTALS = {
    "TOTAL.NO.OF.ACCTS": ("PRI.NO.OF.ACCTS", "SEC.NO.OF.ACCTS"),
    "TOTAL.ACTIVE.ACCTS": ("PRI.ACTIVE.ACCTS", "SEC.ACTIVE.ACCTS"),
    "TOTAL.OVERDUE.ACCTS": ("PRI.OVERDUE.ACCTS", "SEC.OVERDUE.ACCTS"),
    "TOTAL.CURRENT.BALANCE": ("PRI.CURRENT.BALANCE", "SEC.CURRENT.BALANCE"),
    "TOTAL.SANCTIONED.AMOUNT": ("PRI.SANCTIONED.AMOUNT", "SEC.SANCTIONED.AMOUNT"),
    "TOTAL.DISBURSED.AMOUNT": ("PRI.DISBURSED.AMOUNT", "SEC.DISBURSED.AMOUNT"),
    "TOTAL.INSTAL.AMT": ("PRIMARY.INSTAL.AMT", "SEC.INSTAL.AMT"),
}


def to_date(values):
    """Parse dd-mm-yy dates, falling back to pandas' guess if the file uses another format."""
    parsed = pd.to_datetime(values, format="%d-%m-%y", errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(values, dayfirst=True, errors="coerce")
    return parsed


def age_at_disbursal(dob_values, disbursal_values):
    """Borrower age in years on the disbursal date.

    With 2-digit years Python reads 00-68 as 2000-2068, so "01-01-55" becomes
    2055. Any birth date after its disbursal date is moved back 100 years.
    """
    dob = to_date(dob_values)
    disbursal = to_date(disbursal_values)
    in_future = dob > disbursal
    dob[in_future] = dob[in_future] - pd.DateOffset(years=100)
    return (disbursal - dob).dt.days / 365.25


def duration_to_months(values):
    """Convert strings like "1yrs 11mon" to total months (23)."""
    parts = values.astype(str).str.extract(r"(\d+)\s*yrs\s*(\d+)\s*mon")
    return (parts[0].astype(float) * 12 + parts[1].astype(float)).fillna(0)


def cns_band(description):
    """Group a bureau score description into one of CNS_BANDS."""
    description = str(description)
    if description.startswith("No Bureau"):
        return "No History"
    if description.startswith("Not Scored"):
        return "Not Scored"
    band = description.split("-", 1)[-1].strip()  # "C-Very Low Risk" -> "Very Low Risk"
    return band if band in CNS_BANDS else "Not Scored"


def one_hot(series, categories, prefix):
    """One-hot encode against a fixed category list, so the columns never depend on the data."""
    return pd.DataFrame(
        {f"{prefix}_{c.replace(' ', '_')}": (series == c).astype(int) for c in categories},
        index=series.index,
    )


def clean(raw):
    """Turn the raw Kaggle columns into numeric model features plus the target."""
    df = raw.drop(columns=ID_COLUMNS)

    df["AGE.AT.DISBURSAL"] = age_at_disbursal(df["Date.of.Birth"], df["DisbursalDate"])
    df = df.drop(columns=["Date.of.Birth", "DisbursalDate"])

    for col in ["AVERAGE.ACCT.AGE", "CREDIT.HISTORY.LENGTH"]:
        df[col + ".MONTHS"] = duration_to_months(df[col])
    df = df.drop(columns=["AVERAGE.ACCT.AGE", "CREDIT.HISTORY.LENGTH"])

    for total, (pri, sec) in PRI_SEC_TOTALS.items():
        df[total] = df[pri] + df[sec]
        df = df.drop(columns=[pri, sec])

    df["ANY.DELINQUENCY"] = (
        (df["DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"] > 0) | (df["TOTAL.OVERDUE.ACCTS"] > 0)
    ).astype(int)

    employment = df.pop("Employment.Type").fillna("Missing")
    bands = df.pop("PERFORM_CNS.SCORE.DESCRIPTION").map(cns_band)
    df = pd.concat(
        [df, one_hot(employment, EMPLOYMENT_TYPES, "EMP"), one_hot(bands, CNS_BANDS, "CNS")],
        axis=1,
    )

    # Columns with a single value (MobileNo_Avl_Flag is expected to be all 1s)
    constant = [c for c in df.columns if c != TARGET and df[c].nunique() <= 1]
    if constant:
        print(f"Dropping constant columns: {constant}")
        df = df.drop(columns=constant)

    missing_age = df["AGE.AT.DISBURSAL"].isna().sum()
    if missing_age:
        print(f"Filling {missing_age} unparseable ages with the median")
        df["AGE.AT.DISBURSAL"] = df["AGE.AT.DISBURSAL"].fillna(df["AGE.AT.DISBURSAL"].median())

    # Target goes last so the CSVs read naturally
    return df[[c for c in df.columns if c != TARGET] + [TARGET]]


def describe_raw(raw):
    """Print the EDA checks the README's dataset description needs."""
    print(f"Raw data: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    balance = raw[TARGET].value_counts()
    print(f"Class balance: {balance.to_dict()}  (default rate {raw[TARGET].mean():.2%})")
    missing = raw.isna().sum()
    print(f"Missing values: {missing[missing > 0].to_dict() or 'none'}")
    print(raw[["disbursed_amount", "asset_cost", "ltv", "PERFORM_CNS.SCORE"]].describe().round(1))


def main():
    raw = pd.read_csv(RAW_PATH)
    describe_raw(raw)

    data = clean(raw)
    age = data["AGE.AT.DISBURSAL"]
    print(f"\nAge at disbursal: min {age.min():.1f}, median {age.median():.1f}, max {age.max():.1f}")
    print(f"Cleaned data: {data.shape[0]:,} rows x {data.shape[1] - 1} features + target")

    # Stratified subsample keeps the default rate while keeping SVM/KNN fast
    subsample, _ = train_test_split(
        data, train_size=SUBSAMPLE_SIZE, stratify=data[TARGET], random_state=RANDOM_STATE
    )
    train, test = train_test_split(
        subsample, test_size=TEST_SIZE, stratify=subsample[TARGET], random_state=RANDOM_STATE
    )
    for name, part in [("Subsample", subsample), ("Train", train), ("Test", test)]:
        print(f"{name}: {len(part):,} rows, default rate {part[TARGET].mean():.2%}")

    TRAIN_OUT.parent.mkdir(exist_ok=True)
    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)
    features = [c for c in data.columns if c != TARGET]
    FEATURES_OUT.write_text(json.dumps(features, indent=2))

    print(f"\nFeatures ({len(features)}): {features}")
    print(f"Wrote {TRAIN_OUT.relative_to(ROOT)}, {TEST_OUT.relative_to(ROOT)}, {FEATURES_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
