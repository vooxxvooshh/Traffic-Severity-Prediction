import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight

from evaluation import evaluate_model

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def apply_smote(X_train: np.ndarray, y_train) -> tuple[np.ndarray, np.ndarray]:
    # For highly imbalanced multiclass data, SMOTE can help, but it will fail when
    # a class has too few samples to create k-nearest neighbors.
    minority_count = int(np.min(np.bincount(y_train.astype(int))))
    if minority_count < 4:
        print("Skipping SMOTE (minority class too small).")
        return X_train, y_train

    # Use the largest k that is safe given the smallest minority class.
    k = min(5, minority_count - 1)
    smote = SMOTE(random_state=42, k_neighbors=k)
    return smote.fit_resample(X_train, y_train)



def _class_weights(y_train) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def train_baseline_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def run_model_search(X_train, y_train, sample_size: int = 60_000) -> list[dict]:
    if len(y_train) > sample_size:
        idx = np.random.RandomState(42).choice(len(y_train), sample_size, replace=False)
        X_search = X_train[idx]
        y_search = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]
    else:
        X_search = X_train
        y_search = y_train

    X_smote, y_smote = apply_smote(X_search, y_search)
    weight_map = _class_weights(y_smote)


    candidates = []

    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
    rf_params = {
        "n_estimators": [200, 300, 450],
        "max_depth": [10, 14, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }
    candidates.append(("RandomForest + SMOTE", rf, rf_params))


    if HAS_XGB:
        xgb = XGBClassifier(
            objective="multi:softmax",
            num_class=len(np.unique(y_smote)),
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
        xgb_params = {
            "n_estimators": [200, 350, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0, 0.25, 0.5],
        }
        candidates.append(("XGBoost + SMOTE", xgb, xgb_params))

    if HAS_LGBM:
        lgbm = LGBMClassifier(
            objective="multiclass",
            num_class=len(np.unique(y_smote)),
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            class_weight=weight_map,
        )
        lgbm_params = {
            "n_estimators": [250, 400, 600],
            "max_depth": [6, 10, -1],
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [31, 63, 127],
            "subsample": [0.7, 0.85, 1.0],
            "min_child_samples": [5, 10, 20],
        }
        candidates.append(("LightGBM + SMOTE", lgbm, lgbm_params))

    results = []
    for name, estimator, params in candidates:
        print(f"\n--- Tuning {name} ---")
        search = RandomizedSearchCV(
            estimator,
            param_distributions=params,
            n_iter=12,
            cv=5,
            scoring="f1_macro",
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_smote, y_smote)
        results.append(
            {
                "name": name,
                "model": search.best_estimator_,
                "best_params": search.best_params_,
                "cv_f1_macro": float(search.best_score_),
            }
        )
        print(f"Best CV F1 (macro): {search.best_score_:.4f}")
        print("Best params:", search.best_params_)

    return results


def train_best_model(X_train, y_train, search_results: list[dict]):
    X_smote, y_smote = apply_smote(X_train, y_train)
    best = max(search_results, key=lambda r: r["cv_f1_macro"])
    print(f"\nSelected best model: {best['name']} (CV F1 macro={best['cv_f1_macro']:.4f})")
    model = best["model"]
    model.fit(X_smote, y_smote)
    return model, best


def save_artifacts(model, artifacts: dict, model_path: str = "traffic_model.pkl"):
    joblib.dump(model, model_path)
    joblib.dump(artifacts, "model_artifacts.pkl")


def run_full_training_pipeline(data: dict) -> tuple:
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    artifacts = data["artifacts"]

    print("\n--- BASELINE (no SMOTE, original RF) ---")
    baseline = train_baseline_rf(X_train, y_train)
    baseline_metrics = evaluate_model(baseline, X_test, y_test, "baseline_rf")

    scalar_keys = (
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    )
    comparison_rows = [
        {
            "model": "baseline_rf",
            "cv_f1_macro": None,
            **{k: baseline_metrics[k] for k in scalar_keys},
        }
    ]

    print("\n--- HYPERPARAMETER SEARCH (5-fold CV, SMOTE) ---")
    search_results = run_model_search(X_train, y_train)
    best_model, best_info = train_best_model(X_train, y_train, search_results)
    best_metrics = evaluate_model(best_model, X_test, y_test, best_info["name"])

    for result in search_results:
        row = {
            "model": result["name"],
            "cv_f1_macro": result["cv_f1_macro"],
        }
        if result["name"] == best_info["name"]:
            row.update({k: best_metrics[k] for k in scalar_keys})
        comparison_rows.append(row)

    comparison_rows.append(
        {
            "model": best_info["name"] + " (selected)",
            "cv_f1_macro": best_info["cv_f1_macro"],
            **{k: best_metrics[k] for k in scalar_keys},
        }
    )

    artifacts["best_model_name"] = best_info["name"]
    artifacts["best_params"] = best_info["best_params"]
    save_artifacts(best_model, artifacts)

    return best_model, best_metrics, comparison_rows, baseline_metrics
