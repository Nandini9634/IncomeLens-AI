# IncomeLens-AI

## a. Problem statement
Build an end-to-end machine learning classification system that predicts whether a person earns more than 50K per year (`>50K`) using demographic and employment features from the UCI Adult dataset. The project implements 6 required ML classifiers, evaluates each with required metrics, and provides an interactive Streamlit interface for inference and model comparison.

## b. Dataset description
- **Dataset Source:** UCI Machine Learning Repository
- **Dataset Name:** Adult (Census Income)
- **UCI ID:** 2
- **Task Type:** Binary Classification (`<=50K` vs `>50K`)
- **Instances:** 48,842
- **Features:** 14
- **Why this dataset fits assignment constraints:** It satisfies the minimum size constraints (>=500 rows, >=12 features) and is directly importable in Python via `ucimlrepo`.

### Feature columns
`age`, `workclass`, `fnlwgt`, `education`, `education-num`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`

### Class balance analysis
- Class `<=50K` (label `0`): **37,155**
- Class `>50K` (label `1`): **11,687**
- Imbalance ratio (majority:minority): **3.18:1**

Because the dataset is moderately imbalanced, MCC, AUC, and precision-recall behavior were considered alongside raw accuracy.

## c. Models used
All six required models are trained on the **same train/test split**.

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Naive Bayes (GaussianNB)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Comparison table (required metrics)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8507 | 0.9042 | 0.7314 | 0.5941 | 0.6557 | 0.5666 |
| Decision Tree | 0.8621 | 0.9039 | 0.7876 | 0.5804 | 0.6683 | 0.5946 |
| kNN | 0.8442 | 0.8856 | 0.6996 | 0.6116 | 0.6527 | 0.5549 |
| Naive Bayes | 0.5715 | 0.7949 | 0.3517 | 0.9376 | 0.5116 | 0.3497 |
| Random Forest (Ensemble) | 0.8675 | 0.9190 | 0.7979 | 0.5979 | 0.6836 | 0.6120 |
| XGBoost (Ensemble) | 0.8791 | 0.9315 | 0.7923 | 0.6707 | 0.7264 | 0.6532 |

### Observations about model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong baseline with high AUC, but moderate recall indicates missed positive-income samples. |
| Decision Tree | Higher precision than logistic regression but recall is lower, indicating conservative positive predictions. |
| kNN | Reasonable balance, but weaker than tree ensembles on AUC and MCC. |
| Naive Bayes | Very high recall but poor precision/accuracy, generating many false positives due to independence assumptions. |
| Random Forest (Ensemble) | Better balance than single-tree models and improved MCC versus logistic regression/kNN. |
| XGBoost (Ensemble) | Best overall performer across Accuracy, AUC, F1, and MCC on this dataset. |

## Lightweight hyperparameter tuning summary
A lightweight validation-based search (3 candidate sets per model) was applied to tree-based/ensemble models.

| Model | Best Parameters | Validation F1 |
|---|---|---:|
| Decision Tree | `max_depth=10`, `min_samples_split=10`, `min_samples_leaf=3` | 0.6648 |
| Random Forest (Ensemble) | `n_estimators=350`, `max_depth=18`, `min_samples_leaf=1` | 0.6835 |
| XGBoost (Ensemble) | `n_estimators=360`, `learning_rate=0.06`, `max_depth=6`, `subsample=0.9` | 0.7191 |

## Model selection recommendation
- **Best overall model:** XGBoost (highest Accuracy, AUC, F1, MCC)
- **If recall is highest priority:** Naive Bayes (recall 0.9376), but precision trade-off is severe
- **Best practical trade-off:** XGBoost or Random Forest depending on interpretability/performance preference

## Error analysis (XGBoost on held-out test set)
Confusion matrix for XGBoost: `[[7020, 411], [770, 1568]]`

- False Positives (411): some `<=50K` records are predicted as `>50K`
- False Negatives (770): a larger group of true `>50K` is still missed
- Interpretation: model is conservative enough to keep precision high, but further threshold tuning could reduce false negatives

## Repository structure

```text
IncomeLens-AI/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── adult_test_features.csv
│   └── adult_test_with_target.csv
└── model/
    ├── train_models.py
    └── artifacts/
        ├── logistic_regression.joblib
        ├── decision_tree.joblib
        ├── knn.joblib
        ├── naive_bayes.joblib
        ├── random_forest_ensemble.joblib
        ├── xgboost_ensemble.joblib
        ├── metrics_comparison.csv
        ├── metrics_detailed.json
        ├── tuning_summary.json
        └── metadata.json
```

## Streamlit features implemented
1. Dataset upload option (CSV)
2. Model selection dropdown
3. Display of evaluation metrics
4. Confusion matrix and classification report
5. ROC curve and Precision-Recall curve visualization

## Error handling and troubleshooting
- **Uploaded CSV missing required columns**  
  App validates required feature columns from `metadata.json` and stops with a clear error listing missing columns.
- **Uploaded CSV has no target column (`income`)**  
  App still runs prediction and shows precomputed held-out confusion matrix/report/curves instead of failing.
- **Unknown categorical values in uploaded CSV**  
  Handled safely via `OneHotEncoder(handle_unknown=\"ignore\")` in the training pipeline.
- **Missing values in input features**  
  Handled using imputers in preprocessing (`median` for numeric, `most_frequent` for categorical).
- **XGBoost runtime issue on macOS (`libomp` missing)**  
  Install once with `brew install libomp` and run training using:  
  `DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python model/train_models.py`
- **Model artifacts not found**  
  Run training first to regenerate artifacts in `model/artifacts/`, then restart Streamlit.

## Reproducibility
- Random seed: `42`
- Train-test split: `80:20` stratified
- Dataset fetch method: `ucimlrepo.fetch_ucirepo(id=2)`
- Training command:

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python model/train_models.py
```

## Run locally

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit deployment steps
1. Push this folder to GitHub.
2. Open Streamlit Community Cloud.
3. Click `New app`.
4. Select repo/branch and `app.py`.
5. Deploy.

