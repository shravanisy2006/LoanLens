
# ============================================================
# LOANLENS AI - IMPROVED LOAN DEFAULT PREDICTION
# Cross-validation, hyperparameter tuning, threshold optimization
# ============================================================

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_predict,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "loan.csv"
MODEL_DIR = BASE_DIR / "models"
ASSET_DIR = BASE_DIR / "assets"

MODEL_DIR.mkdir(exist_ok=True)
ASSET_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

print("\n" + "=" * 65)
print("LOANLENS AI - MODEL OPTIMIZATION")
print("=" * 65)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH, low_memory=False)

numeric_features = [
    "loan_amnt",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "total_acc",
    "int_rate",
    "revol_util",
]

categorical_features = [
    "term",
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "emp_length",
    "application_type",
]

features = numeric_features + categorical_features

required_columns = ["loan_status"] + features
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing dataset columns: {missing_columns}"
    )

# Use only loans with known final outcomes.
df = df[
    df["loan_status"].isin(["Fully Paid", "Charged Off"])
].copy()

# Charged Off = 1; Fully Paid = 0.
df["default"] = (
    df["loan_status"] == "Charged Off"
).astype(int)

X = df[features].copy()
y = df["default"].copy()

# Convert interest rate and revolving utilization to numbers.
for column in ["int_rate", "revol_util"]:
    X[column] = pd.to_numeric(
        X[column]
        .astype(str)
        .str.replace("%", "", regex=False),
        errors="coerce",
    )

for column in numeric_features:
    X[column] = pd.to_numeric(
        X[column], errors="coerce"
    )

for column in categorical_features:
    X[column] = (
        X[column]
        .astype(object)
        .replace(["", "nan", "None", "<NA>"], np.nan)
    )

    X[column] = X[column].where(
        pd.notna(X[column]), np.nan
    )

print(f"Resolved loans: {len(X):,}")
print(f"Number of features: {len(features)}")
print(f"Default cases: {int(y.sum()):,}")
print(f"Default rate: {y.mean():.2%}")


# ------------------------------------------------------------
# 3. TRAIN-TEST SPLIT
# ------------------------------------------------------------

# Keep this test set separate from tuning and threshold selection.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

print(f"\nTraining records: {len(X_train):,}")
print(f"Test records: {len(X_test):,}")
print(f"Cross-validation folds: {N_SPLITS}")


# ------------------------------------------------------------
# 4. PREPROCESSING
# ------------------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])


# ------------------------------------------------------------
# 5. DEFINE MODELS AND SEARCH GRIDS
# ------------------------------------------------------------

model_configs = {
    "Logistic Regression": {
        "estimator": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "params": {
            "model__C": [0.1, 1.0, 5.0],
        },
    },

    "Decision Tree": {
        "estimator": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "params": {
            "model__max_depth": [6, 10, 14],
            "model__min_samples_leaf": [10, 25],
        },
    },

    "Random Forest": {
        "estimator": RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [10, 16],
            "model__min_samples_leaf": [5, 10],
        },
    },
}


# ------------------------------------------------------------
# 6. TUNE EACH MODEL
# ------------------------------------------------------------

best_models = {}
thresholds = {}
cv_rows = {}

for name, config in model_configs.items():

    print("\n" + "-" * 60)
    print(f"Tuning {name}...")
    print("-" * 60)

    pipeline = Pipeline([
        ("preprocessing", clone(preprocessor)),
        ("model", config["estimator"]),
    ])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=config["params"],
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    best_models[name] = best_pipeline

    # Out-of-fold probabilities use training data only.
    oof_probabilities = cross_val_predict(
        clone(best_pipeline),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    # Choose a threshold that maximizes training OOF F1.
    precision, recall, candidate_thresholds = (
        precision_recall_curve(y_train, oof_probabilities)
    )

    if len(candidate_thresholds):
        f1_values = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )

        best_index = int(np.argmax(f1_values))
        threshold = float(candidate_thresholds[best_index])
        oof_f1 = float(f1_values[best_index])
    else:
        threshold = 0.5
        oof_f1 = 0.0

    thresholds[name] = threshold

    cv_auc = float(search.best_score_)
    oof_auc = float(
        roc_auc_score(y_train, oof_probabilities)
    )
    oof_ap = float(
        average_precision_score(y_train, oof_probabilities)
    )

    # Combined selection score: balances ranking and F1.
    # This is a model-selection heuristic, not a standard metric.
    combined_score = 0.5 * cv_auc + 0.5 * oof_f1

    cv_rows[name] = {
        "Model": name,
        "Mean CV ROC-AUC": cv_auc,
        "OOF ROC-AUC": oof_auc,
        "OOF Average Precision": oof_ap,
        "OOF F1": oof_f1,
        "Threshold": threshold,
        "Combined Score": combined_score,
        "Best Parameters": str(search.best_params_),
    }

    print("Best parameters:", search.best_params_)
    print(f"Mean CV ROC-AUC: {cv_auc:.4f}")
    print(f"OOF F1-score: {oof_f1:.4f}")
    print(f"Selected threshold: {threshold:.4f}")
    print(f"Combined selection score: {combined_score:.4f}")


# ------------------------------------------------------------
# 7. SELECT BEST MODEL
# ------------------------------------------------------------

cv_comparison = pd.DataFrame(list(cv_rows.values()))

cv_comparison = cv_comparison.sort_values(
    "Combined Score",
    ascending=False,
).reset_index(drop=True)

best_model_name = cv_comparison.iloc[0]["Model"]
best_model = best_models[best_model_name]
best_threshold = thresholds[best_model_name]

print("\n" + "=" * 65)
print("CROSS-VALIDATION COMPARISON")
print("=" * 65)

print(
    cv_comparison[
        [
            "Model",
            "Mean CV ROC-AUC",
            "OOF F1",
            "Threshold",
            "Combined Score",
        ]
    ].round(4).to_string(index=False)
)

print(f"\nSelected model: {best_model_name}")
print(f"Selected threshold: {best_threshold:.4f}")


# ------------------------------------------------------------
# 8. EVALUATE ALL TUNED MODELS ON THE SAME TEST SET
# ------------------------------------------------------------

test_rows = []
test_probabilities_by_model = {}
test_predictions_by_model = {}

for name, pipeline in best_models.items():

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    threshold = thresholds[name]
    predictions = (probabilities >= threshold).astype(int)

    test_probabilities_by_model[name] = probabilities
    test_predictions_by_model[name] = predictions

    test_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Average Precision": average_precision_score(
            y_test, probabilities
        ),
        "Threshold": threshold,
    })

test_comparison = pd.DataFrame(test_rows)

# Put the selected model first.
test_comparison["_selected"] = (
    test_comparison["Model"] == best_model_name
)
test_comparison = (
    test_comparison.sort_values(
        "_selected", ascending=False
    )
    .drop(columns="_selected")
    .reset_index(drop=True)
)

print("\n" + "=" * 65)
print("FINAL HELD-OUT TEST RESULTS")
print("=" * 65)
print(test_comparison.round(4).to_string(index=False))

selected_test_row = test_comparison[
    test_comparison["Model"] == best_model_name
].iloc[0]

print("\nSelected model test metrics:")
for metric in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC",
    "Average Precision",
]:
    print(f"{metric:20s}: {selected_test_row[metric]:.4f}")


# ------------------------------------------------------------
# 9. SAVE CONFUSION MATRIX
# ------------------------------------------------------------

selected_predictions = test_predictions_by_model[best_model_name]

cm = confusion_matrix(
    y_test,
    selected_predictions,
    labels=[0, 1],
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fully Paid", "Charged Off"],
)

fig, ax = plt.subplots(figsize=(7, 5))
display.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title(
    f"{best_model_name} — Test Confusion Matrix\n"
    f"Threshold = {best_threshold:.3f}"
)
fig.tight_layout()

fig.savefig(
    ASSET_DIR / "best_model_confusion_matrix.png",
    dpi=160,
    bbox_inches="tight",
)
plt.close(fig)


# ------------------------------------------------------------
# 10. SAVE PRECISION-RECALL CURVE
# ------------------------------------------------------------

selected_probabilities = test_probabilities_by_model[best_model_name]

precision, recall, _ = precision_recall_curve(
    y_test,
    selected_probabilities,
)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recall, precision)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — Held-Out Test Set")
ax.grid(alpha=0.3)
fig.tight_layout()

fig.savefig(
    ASSET_DIR / "precision_recall_curve.png",
    dpi=160,
    bbox_inches="tight",
)
plt.close(fig)


# ------------------------------------------------------------
# 11. SAVE RESULTS AND MODEL
# ------------------------------------------------------------

cv_comparison.to_csv(
    MODEL_DIR / "cross_validation_comparison.csv",
    index=False,
)

# Keep all three models in this file for the frontend comparison.
test_comparison.to_csv(
    MODEL_DIR / "model_comparison.csv",
    index=False,
)

test_comparison[
    test_comparison["Model"] == best_model_name
].to_csv(
    MODEL_DIR / "final_test_metrics.csv",
    index=False,
)

joblib.dump(
    best_model,
    MODEL_DIR / "best_model.joblib",
)

metadata = {
    "best_model": best_model_name,
    "features": features,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "target_mapping": {
        "Fully Paid": 0,
        "Charged Off": 1,
    },
    "selection_metric": (
        "Combined score: 0.5 * mean CV ROC-AUC + 0.5 * OOF F1"
    ),
    "threshold_metric": "Out-of-fold training F1-score",
    "decision_threshold": float(best_threshold),
    "cv_folds": N_SPLITS,
    "random_state": RANDOM_STATE,
    "test_size": 0.20,
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "test_metrics": {
        key: float(selected_test_row[key])
        for key in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "Average Precision",
        ]
    },
}

with open(
    MODEL_DIR / "model_metadata.json", "w"
) as file:
    json.dump(metadata, file, indent=4)

print("\n" + "=" * 65)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 65)
print("Selected model:", best_model_name)
print(f"Decision threshold: {best_threshold:.4f}")
print("Saved model:", MODEL_DIR / "best_model.joblib")
print("Saved metadata:", MODEL_DIR / "model_metadata.json")
print("CV results:", MODEL_DIR / "cross_validation_comparison.csv")
print("Test comparison:", MODEL_DIR / "model_comparison.csv")
print("Test confusion matrix:", ASSET_DIR / "best_model_confusion_matrix.png")
print("Precision-recall curve:", ASSET_DIR / "precision_recall_curve.png")
