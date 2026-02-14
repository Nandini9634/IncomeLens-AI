from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier


RANDOM_STATE = 42


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )


def get_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=12),
        "kNN": KNeighborsClassifier(n_neighbors=11),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=350,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray) -> tuple[dict, dict]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_proba)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "AveragePrecision": average_precision_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    extra = {
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
        },
        "pr_curve": {
            "precision": pr_precision.tolist(),
            "recall": pr_recall.tolist(),
        },
    }
    return metrics, extra


def tune_tree_ensemble_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    preprocessor: ColumnTransformer,
) -> dict[str, dict]:
    X_subtrain, X_valid, y_subtrain, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    search_space = {
        "Decision Tree": [
            {"max_depth": 8, "min_samples_split": 12, "min_samples_leaf": 4},
            {"max_depth": 10, "min_samples_split": 10, "min_samples_leaf": 3},
            {"max_depth": 12, "min_samples_split": 10, "min_samples_leaf": 2},
        ],
        "Random Forest (Ensemble)": [
            {"n_estimators": 200, "max_depth": 14, "min_samples_leaf": 2},
            {"n_estimators": 300, "max_depth": 16, "min_samples_leaf": 2},
            {"n_estimators": 350, "max_depth": 18, "min_samples_leaf": 1},
        ],
        "XGBoost (Ensemble)": [
            {"n_estimators": 280, "learning_rate": 0.07, "max_depth": 5, "subsample": 0.85},
            {"n_estimators": 320, "learning_rate": 0.08, "max_depth": 6, "subsample": 0.9},
            {"n_estimators": 360, "learning_rate": 0.06, "max_depth": 6, "subsample": 0.9},
        ],
    }

    tuned: dict[str, dict] = {}
    for model_name, candidates in search_space.items():
        best_params = candidates[0]
        best_f1 = -1.0

        for params in candidates:
            if model_name == "Decision Tree":
                estimator = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
            elif model_name == "Random Forest (Ensemble)":
                estimator = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
            else:
                estimator = XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    colsample_bytree=0.9,
                    **params,
                )

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", estimator),
                ]
            )
            pipeline.fit(X_subtrain, y_subtrain)
            y_valid_pred = pipeline.predict(X_valid)
            current_f1 = f1_score(y_valid, y_valid_pred)

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_params = params

        tuned[model_name] = {
            "best_params": best_params,
            "validation_f1": round(float(best_f1), 6),
            "n_candidates_evaluated": len(candidates),
        }
    return tuned


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "model" / "artifacts"
    data_dir = project_root / "data"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = fetch_ucirepo(id=2)  # Adult
    X = dataset.data.features.copy()
    y_raw = dataset.data.targets.copy().iloc[:, 0]

    X = X.replace("?", np.nan)
    y_raw = y_raw.astype(str).str.strip().str.replace(".", "", regex=False)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(X)
    tuning_summary = tune_tree_ensemble_models(X_train, y_train, preprocessor)
    models = get_models()
    models["Decision Tree"] = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        **tuning_summary["Decision Tree"]["best_params"],
    )
    models["Random Forest (Ensemble)"] = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **tuning_summary["Random Forest (Ensemble)"]["best_params"],
    )
    models["XGBoost (Ensemble)"] = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        colsample_bytree=0.9,
        **tuning_summary["XGBoost (Ensemble)"]["best_params"],
    )

    metrics_rows: list[dict] = []
    detailed_results: dict[str, dict] = {}

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        metrics, extra = evaluate_model(pipeline, X_test, y_test)

        metrics_row = {"ML Model Name": model_name}
        metrics_row.update({k: round(v, 4) for k, v in metrics.items() if k != "AveragePrecision"})
        metrics_rows.append(metrics_row)

        detailed_results[model_name] = {
            "metrics": {k: float(round(v, 6)) for k, v in metrics.items()},
            "confusion_matrix": extra["confusion_matrix"],
            "classification_report": extra["classification_report"],
            "roc_curve": extra["roc_curve"],
            "pr_curve": extra["pr_curve"],
        }

        model_path = artifacts_dir / f"{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl.gz"
        with gzip.open(model_path, "wb") as f:
            pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(artifacts_dir / "metrics_comparison.csv", index=False)

    with open(artifacts_dir / "metrics_detailed.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)

    class_counts = pd.Series(y).value_counts().sort_index()
    imbalance_ratio = float(class_counts.max() / class_counts.min())
    metadata = {
        "dataset_name": "UCI Adult Income",
        "dataset_id": 2,
        "n_instances": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_columns": X.columns.tolist(),
        "target_column": "income",
        "target_mapping": {
            "0": str(label_encoder.classes_[0]),
            "1": str(label_encoder.classes_[1]),
        },
        "class_distribution": {
            "0": int(class_counts.iloc[0]),
            "1": int(class_counts.iloc[1]),
        },
        "imbalance_ratio_majority_to_minority": round(imbalance_ratio, 4),
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
    }
    with open(artifacts_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(artifacts_dir / "tuning_summary.json", "w", encoding="utf-8") as f:
        json.dump(tuning_summary, f, indent=2)

    test_features = X_test.copy()
    test_features.to_csv(data_dir / "adult_test_features.csv", index=False)

    test_with_target = X_test.copy()
    test_with_target["income"] = y_test
    test_with_target.to_csv(data_dir / "adult_test_with_target.csv", index=False)

    print("Training complete.")
    print(f"Artifacts saved to: {artifacts_dir}")


if __name__ == "__main__":
    main()
