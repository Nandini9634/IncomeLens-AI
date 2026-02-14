from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    import joblib
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib==1.5.3"])
    import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

st.set_page_config(page_title="Adult Income Classifier", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(120deg, #f8fafc 0%, #eef6ff 50%, #f9f4ff 100%);
    }
    .title-box {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        background: #0f172a;
        color: #e2e8f0;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_models() -> dict[str, object]:
    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "kNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
        "XGBoost (Ensemble)": "xgboost_ensemble.joblib",
    }
    loaded = {}
    for model_name, filename in model_files.items():
        loaded[model_name] = joblib.load(ARTIFACTS_DIR / filename)
    return loaded


metadata = load_json(ARTIFACTS_DIR / "metadata.json")
detailed_metrics = load_json(ARTIFACTS_DIR / "metrics_detailed.json")
metrics_df = pd.read_csv(ARTIFACTS_DIR / "metrics_comparison.csv")
models = load_models()

st.markdown('<div class="title-box"><h1>UCI Adult Income Classification Lab</h1><p>Predict whether annual income is >50K using six ML classifiers.</p></div>', unsafe_allow_html=True)

left_col, right_col = st.columns([1.1, 1.2], gap="large")

with left_col:
    st.subheader("1) Dataset Upload (CSV)")
    st.caption("Use `data/adult_test_features.csv` or `data/adult_test_with_target.csv` for quick demo.")
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])

    if st.button("Load local sample CSV"):
        sample_df = pd.read_csv(DATA_DIR / "adult_test_with_target.csv")
        st.session_state["uploaded_df"] = sample_df

    if uploaded_file is not None:
        st.session_state["uploaded_df"] = pd.read_csv(uploaded_file)

    df = st.session_state.get("uploaded_df")

    if df is not None:
        st.write("Preview:")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("Upload a CSV or click 'Load local sample CSV' to start prediction.")

with right_col:
    st.subheader("2) Model Selection")
    model_name = st.selectbox("Choose a classifier", list(models.keys()))

    st.subheader("3) Evaluation Metrics (Held-out Test Set)")
    selected_metrics = detailed_metrics[model_name]["metrics"]
    metric_cols = st.columns(3)
    metric_cols[0].metric("Accuracy", f"{selected_metrics['Accuracy']:.4f}")
    metric_cols[1].metric("AUC", f"{selected_metrics['AUC']:.4f}")
    metric_cols[2].metric("Precision", f"{selected_metrics['Precision']:.4f}")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Recall", f"{selected_metrics['Recall']:.4f}")
    metric_cols[1].metric("F1", f"{selected_metrics['F1']:.4f}")
    metric_cols[2].metric("MCC", f"{selected_metrics['MCC']:.4f}")

st.markdown("---")
st.subheader("4) Confusion Matrix / Classification Report")

selected_model = models[model_name]
input_df = st.session_state.get("uploaded_df")

if input_df is None:
    cm = detailed_metrics[model_name]["confusion_matrix"]
    report = detailed_metrics[model_name]["classification_report"]
    roc_payload = detailed_metrics[model_name]["roc_curve"]
    pr_payload = detailed_metrics[model_name]["pr_curve"]
    roc_fpr = roc_payload["fpr"]
    roc_tpr = roc_payload["tpr"]
    pr_recall = pr_payload["recall"]
    pr_precision = pr_payload["precision"]
    roc_auc_value = selected_metrics["AUC"]
    pr_auc_value = selected_metrics["AveragePrecision"]
    st.caption("Showing precomputed results from the held-out test split.")
else:
    feature_columns = metadata["feature_columns"]
    missing_cols = [c for c in feature_columns if c not in input_df.columns]
    if missing_cols:
        st.error(f"Missing required feature columns: {missing_cols}")
        st.stop()

    X_infer = input_df[feature_columns].copy()
    y_pred = selected_model.predict(X_infer)
    y_score = selected_model.predict_proba(X_infer)[:, 1]

    st.write("Predictions (first 20 rows):")
    pred_df = input_df.copy()
    pred_df["pred_income_class"] = y_pred
    pred_df["pred_income_label"] = pred_df["pred_income_class"].map(
        {0: metadata["target_mapping"]["0"], 1: metadata["target_mapping"]["1"]}
    )
    st.dataframe(pred_df.head(20), use_container_width=True)

    if "income" in input_df.columns:
        y_true = input_df["income"]
        cm = confusion_matrix(y_true, y_pred).tolist()
        report = classification_report(y_true, y_pred, output_dict=True)
        roc_fpr, roc_tpr, _ = roc_curve(y_true, y_score)
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_score)
        roc_auc_value = auc(roc_fpr, roc_tpr)
        pr_auc_value = average_precision_score(y_true, y_score)
        st.caption("Confusion matrix/report computed from uploaded CSV containing target column `income`.")
    else:
        cm = detailed_metrics[model_name]["confusion_matrix"]
        report = detailed_metrics[model_name]["classification_report"]
        roc_payload = detailed_metrics[model_name]["roc_curve"]
        pr_payload = detailed_metrics[model_name]["pr_curve"]
        roc_fpr = roc_payload["fpr"]
        roc_tpr = roc_payload["tpr"]
        pr_recall = pr_payload["recall"]
        pr_precision = pr_payload["precision"]
        roc_auc_value = selected_metrics["AUC"]
        pr_auc_value = selected_metrics["AveragePrecision"]
        st.caption("Uploaded CSV has no `income` column, showing precomputed held-out test results.")

cm_col, report_col = st.columns([1, 1.2], gap="large")

with cm_col:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with report_col:
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

st.markdown("---")
st.subheader("5) ROC Curve and Precision-Recall Curve")

roc_col, pr_col = st.columns(2, gap="large")

with roc_col:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(roc_fpr, roc_tpr, color="#2563eb", linewidth=2, label=f"AUC = {roc_auc_value:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1.5)
    ax.set_title(f"{model_name} - ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    st.pyplot(fig)

with pr_col:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(pr_recall, pr_precision, color="#16a34a", linewidth=2, label=f"AP = {pr_auc_value:.4f}")
    ax.set_title(f"{model_name} - Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    st.pyplot(fig)

st.markdown("---")
st.subheader("Model Comparison Table")
st.dataframe(metrics_df, use_container_width=True)

st.caption(
    f"Dataset: {metadata['dataset_name']} (ID {metadata['dataset_id']}) | "
    f"Instances: {metadata['n_instances']} | Features: {metadata['n_features']}"
)
