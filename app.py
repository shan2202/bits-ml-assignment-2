"""Streamlit app: evaluate the saved models on an uploaded test CSV (no retraining)."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from model.train_models import TARGET, compute_metrics, positive_scores

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"
FEATURES = json.loads((MODEL_DIR / "feature_columns.json").read_text())

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
CLASS_NAMES = ["No default", "Default"]

# Each model keeps the same colour in every chart
MODEL_COLORS = {
    "Logistic Regression": "#2a78d6",
    "Decision Tree": "#eb6834",
    "KNN": "#1baf7a",
    "Naive Bayes": "#eda100",
    "Support Vector Machine": "#e87ba4",
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CM_CMAP = LinearSegmentedColormap.from_list("loan_blues", ["#cde2fb", "#6da7ec", "#1c5cab"])

METRIC_HELP = {
    "Accuracy": "Share of all borrowers classified correctly. Misleading here: the baseline of always predicting 'no default' already scores ~78%, and the arrow shows the difference from it.",
    "AUC": "How well the model ranks defaulters above non-defaulters (0.5 = random, 1 = perfect).",
    "Precision": "Of the borrowers flagged as defaulters, the share who really defaulted.",
    "Recall": "Of the borrowers who really defaulted, the share the model caught.",
    "F1": "Harmonic mean of precision and recall.",
    "MCC": "Matthews correlation: balanced score for imbalanced classes (0 = no better than chance, 1 = perfect).",
}


@st.cache_resource
def load_model(name):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


@st.cache_data
def evaluate_all(X, y):
    """Metrics table and positive-class scores for every saved model on the uploaded data."""
    rows, scores = {}, {}
    for name in MODEL_FILES:
        model = load_model(name)
        rows[name] = compute_metrics(model, X, y)
        scores[name] = positive_scores(model, X)
    return pd.DataFrame(rows).T, scores


def validate(df):
    """List the problems that stop the uploaded CSV from being evaluated."""
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        shown = ", ".join(f"`{c}`" for c in missing[:8]) + (" …" if len(missing) > 8 else "")
        return [f"Missing {len(missing)} required column(s): {shown}"]

    problems = []
    non_numeric = [c for c in FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        problems.append(f"These columns must be numeric: {', '.join(non_numeric)}")
    if df[FEATURES + [TARGET]].isna().any().any():
        problems.append("Some required columns have empty values.")
    labels = set(df[TARGET].dropna().unique())
    if not labels <= {0, 1}:
        problems.append(f"`{TARGET}` must contain only 0 and 1.")
    elif len(labels) < 2:
        problems.append(f"`{TARGET}` needs both classes (0 and 1) to compute AUC and MCC.")
    return problems


def new_figure(width, height):
    """Matplotlib figure with the app's quiet chart styling."""
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def confusion_matrix_figure(cm):
    fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=CM_CMAP,
        cbar=False,
        linewidths=2,
        linecolor=SURFACE,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        annot_kws={"fontsize": 12},
        ax=ax,
    )
    ax.set_xlabel("Predicted", color=INK_2, fontsize=9)
    ax.set_ylabel("Actual", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    return fig


def roc_figure(y, scores, aucs):
    """ROC curve for each model in scores, plus the chance diagonal."""
    fig, ax = new_figure(5.5, 4.2)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color=MUTED, label="Chance (AUC 0.500)")
    for name, model_scores in scores.items():
        fpr, tpr, _ = roc_curve(y, model_scores)
        ax.plot(fpr, tpr, linewidth=2, color=MODEL_COLORS[name], label=f"{name} (AUC {aucs[name]:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate (good borrowers flagged)", color=INK_2, fontsize=9)
    ax.set_ylabel("True positive rate (defaulters caught)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK_2)
    return fig


def mcc_figure(table):
    """Horizontal bars of MCC per model, best at the top."""
    mcc = table["MCC"].sort_values()
    fig, ax = new_figure(5.5, 2.8)
    ax.barh(mcc.index, mcc.values, height=0.55, color=[MODEL_COLORS[n] for n in mcc.index])
    ax.axvline(0, color=AXIS, linewidth=1)
    for i, value in enumerate(mcc.values):
        offset = 0.004 if value >= 0 else -0.004
        ax.text(value + offset, i, f"{value:.3f}", va="center", ha="left" if value >= 0 else "right",
                fontsize=8, color=INK_2)
    ax.set_xlim(min(0, mcc.min()) - 0.04, mcc.max() + 0.04)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", colors=INK, labelsize=9, length=0)
    ax.set_xlabel("MCC (0 = no better than chance)", color=INK_2, fontsize=9)
    return fig


st.set_page_config(page_title="Vehicle Loan Default Model Evaluator", page_icon="🛵", layout="wide")

with st.sidebar:
    st.header("About")
    st.markdown(
        "Predicts whether a vehicle-loan borrower will **default on the first EMI**, "
        "using loan details, KYC flags and credit bureau history.\n\n"
        "**Data:** [L&T Vehicle Loan Default Prediction]"
        "(https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction), "
        "Kaggle. 233,154 loans, 21.7% default.\n\n"
        "**Models:** five scikit-learn pipelines trained on a 12,000-row stratified split "
        "on the BITS Virtual Lab. This app only loads them: nothing is retrained on your upload."
    )
    st.divider()
    st.subheader("Need test data?")
    st.caption("The 3,000-row held-out split used in the README. Download it, then upload it on the right.")
    st.download_button(
        "Download test_data.csv",
        data=SAMPLE_CSV.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )

st.title("Vehicle Loan Default: Model Evaluator")
st.caption("Upload labelled test data, pick a model, and see how well it separates defaulters from good borrowers.")

upload_col, model_col = st.columns([2, 1])
with upload_col:
    uploaded = st.file_uploader(
        "1. Upload test data (CSV)",
        type="csv",
        help=f"Needs the {len(FEATURES)} feature columns and `{TARGET}`, as in test_data.csv.",
    )
with model_col:
    model_name = st.selectbox("2. Choose a model", list(MODEL_FILES))

if uploaded is None:
    st.info("Upload a CSV to start. You can download a ready-made test file from the sidebar.")
    with st.expander(f"Columns the CSV must contain ({len(FEATURES) + 1})"):
        st.write(", ".join(f"`{c}`" for c in FEATURES + [TARGET]))
    st.stop()

try:
    data = pd.read_csv(uploaded)
except Exception as err:  # malformed or non-CSV upload
    st.error(f"Could not read the file as CSV: {err}")
    st.stop()

problems = validate(data)
if problems:
    for problem in problems:
        st.error(problem)
    st.stop()

X = data[FEATURES]
y = data[TARGET].astype(int)
all_metrics, all_scores = evaluate_all(X, y)

model = load_model(model_name)
y_pred = model.predict(X)
metrics = all_metrics.loc[model_name]
majority_accuracy = (y == 0).mean()

st.success(f"Loaded {len(data):,} rows: {int(y.sum()):,} defaults ({y.mean():.1%}).")

tab_model, tab_compare, tab_predictions = st.tabs(
    [f"📊 {model_name}", "⚖️ Compare all models", "📄 Predictions"]
)

with tab_model:
    # Two rows of three, so the values are not cut off on narrower screens
    metric_cols = st.columns(3) + st.columns(3)
    for col, (label, value) in zip(metric_cols, metrics.items()):
        delta = None
        if label == "Accuracy":
            delta = f"{value - majority_accuracy:+.4f} vs baseline"
        col.metric(label, f"{value:.4f}", delta=delta, help=METRIC_HELP[label])

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    cm_col, report_col = st.columns(2)
    with cm_col:
        st.subheader("Confusion matrix")
        fig = confusion_matrix_figure(cm)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            f"Caught **{tp:,}** of {tp + fn:,} defaulters and wrongly flagged "
            f"**{fp:,}** of {tn + fp:,} good borrowers."
        )
    with report_col:
        st.subheader("Classification report")
        report = pd.DataFrame(
            classification_report(y, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
                                  output_dict=True, zero_division=0)
        ).T.drop(index="accuracy", errors="ignore")
        report["support"] = report["support"].astype(int)
        st.dataframe(
            report,
            column_config={
                c: st.column_config.NumberColumn(format="%.4f") for c in ["precision", "recall", "f1-score"]
            },
        )
        if tp + fp == 0:
            st.warning("This model never predicts a default on this data, so its precision, recall and F1 are 0.")

    st.subheader("ROC curve")
    fig = roc_figure(y, {model_name: all_scores[model_name]}, all_metrics["AUC"])
    st.pyplot(fig)
    plt.close(fig)

with tab_compare:
    st.subheader("All five models on the uploaded data")
    best = all_metrics.idxmax()
    st.caption(
        f"Best AUC: **{best['AUC']}** · best recall: **{best['Recall']}** · best MCC: **{best['MCC']}**"
    )
    st.dataframe(
        all_metrics.rename_axis("Model"),
        column_config={c: st.column_config.NumberColumn(format="%.4f") for c in all_metrics.columns},
    )
    roc_col, mcc_col = st.columns(2)
    with roc_col:
        fig = roc_figure(y, all_scores, all_metrics["AUC"])
        st.pyplot(fig)
        plt.close(fig)
    with mcc_col:
        fig = mcc_figure(all_metrics)
        st.pyplot(fig)
        plt.close(fig)

with tab_predictions:
    score_label = "default_probability" if hasattr(model, "predict_proba") else "svm_decision_score"
    predictions = pd.DataFrame(
        {"actual": y, "predicted": y_pred, score_label: all_scores[model_name]}
    )
    predictions["correct"] = predictions["actual"] == predictions["predicted"]
    st.dataframe(predictions, column_config={score_label: st.column_config.NumberColumn(format="%.4f")})
    st.download_button(
        f"Download {model_name} predictions",
        data=predictions.to_csv(index=False).encode(),
        file_name=f"predictions_{MODEL_FILES[model_name].removesuffix('.joblib')}.csv",
        mime="text/csv",
    )
