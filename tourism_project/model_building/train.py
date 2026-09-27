'''Tune on training folds, log every trial to MLflow, then evaluate one locked holdout.'''

import argparse
import json
import os
import platform
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    classification_report, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from tourism_project.model_building.schema import (CATEGORICAL, DATA, DEPLOYMENT, DERIVED,
    FEATURES, NUMERIC, PROJECT, REPORTS, SEED, PrepareFeatures, sha256, write_json)


def build_pipeline(estimator):
    # Every fold fits its own imputation and encoding; no holdout statistics are reused.
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocess = ColumnTransformer([("numeric", numeric, NUMERIC + DERIVED),
                                   ("categorical", categorical, CATEGORICAL)])
    return Pipeline([("features", PrepareFeatures()), ("preprocess", preprocess), ("model", estimator)])


def metrics(y, probability, threshold):
    predicted = (probability >= threshold).astype(int)
    return {
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)), "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "f2": float(fbeta_score(y, predicted, beta=2, zero_division=0)),
        "brier_score": float(brier_score_loss(y, probability)),
    }


def train(split_dir=DATA / "splits"):
    started = time.monotonic()
    split_dir = Path(split_dir)
    manifest = json.loads((split_dir / "split_manifest.json").read_text())
    # Validate the exact files downloaded by the training job before trusting the artifact.
    for name, expected_hash in manifest["file_hashes"].items():
        if sha256(split_dir / name) != expected_hash:
            raise ValueError(f"Split artifact checksum failed: {name}")
    X_train, X_test = [pd.read_csv(split_dir / f"X_{part}.csv") for part in ["train", "test"]]
    y_train, y_test = [pd.read_csv(split_dir / f"y_{part}.csv").iloc[:, 0] for part in ["train", "test"]]
    g_train, g_test = [pd.read_csv(split_dir / f"groups_{part}.csv", dtype=str).iloc[:, 0] for part in ["train", "test"]]
    if set(g_train) & set(g_test):
        raise ValueError("Train/test profiles overlap.")
    REPORTS.mkdir(parents=True, exist_ok=True)
    DEPLOYMENT.mkdir(parents=True, exist_ok=True)
    cv = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(X_train, y_train, g_train))
    for fit_index, validation_index in cv:
        assert not (set(g_train.iloc[fit_index]) & set(g_train.iloc[validation_index]))
    mlflow.set_tracking_uri((PROJECT / "mlruns").resolve().as_uri())
    mlflow.set_experiment("visit-with-us-precontact")
    scoring = {"ap": "average_precision", "roc_auc": "roc_auc", "recall": "recall", "f1": "f1"}
    candidates = {
        "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=1), {
            "model__n_estimators": [160, 280], "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1, 3], "model__class_weight": ["balanced"],
        }),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED), {
            "model__n_estimators": [120, 200], "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3], "model__min_samples_leaf": [5],
        }),
    }
    all_trials, comparison, fitted = [], [], {}
    with mlflow.start_run(run_name="grouped-precontact-comparison") as parent:
        mlflow.log_params({"seed": SEED, "selection_metric": "average_precision", "cv_folds": 3,
                          "decision_metric": "F2 on training out-of-fold predictions", "prediction_stage": "pre-contact",
                          "train_rows": len(X_train), "test_rows": len(X_test), "source_sha256": manifest["source_sha256"]})
        mlflow.set_tags({"source_commit": os.environ.get("GITHUB_SHA", "local-execution"),
                         "python": platform.python_version(), "sklearn": sklearn.__version__})
        baseline = cross_validate(build_pipeline(DummyClassifier(strategy="prior")), X_train, y_train,
                                  cv=cv, scoring=scoring, n_jobs=1)
        comparison.append({"model": "Dummy (prior)", "cv_average_precision": float(baseline["test_ap"].mean()),
                           "cv_ap_std": float(baseline["test_ap"].std()), "cv_roc_auc": float(baseline["test_roc_auc"].mean())})
        for name, (estimator, grid) in candidates.items():
            search = GridSearchCV(build_pipeline(estimator), grid, scoring=scoring, refit="ap", cv=cv,
                                  n_jobs=1, error_score="raise", return_train_score=True)
            search.fit(X_train, y_train)
            fitted[name] = search
            for index, parameters in enumerate(search.cv_results_["params"]):
                with mlflow.start_run(run_name=f"{name}-trial-{index + 1:02d}", nested=True) as child:
                    mlflow.log_params({**parameters, "algorithm": name, "random_state": SEED})
                    # Every tuned combination and every held-out fold score is retained.
                    results = {key: float(value[index]) for key, value in search.cv_results_.items()
                               if key.startswith(("mean_", "std_", "split")) and "time" not in key}
                    mlflow.log_metrics(results)
                    all_trials.append({"model": name, "trial": index + 1, "run_id": child.info.run_id,
                                       "parameters": json.dumps(parameters, sort_keys=True), **results})
            index = search.best_index_
            comparison.append({"model": name, "cv_average_precision": float(search.best_score_),
                               "cv_ap_std": float(search.cv_results_["std_test_ap"][index]),
                               "cv_roc_auc": float(search.cv_results_["mean_test_roc_auc"][index])})
            print(f"{name}: best grouped CV AP={search.best_score_:.4f}; parameters={search.best_params_}", flush=True)
        comparison_frame = pd.DataFrame(comparison).sort_values("cv_average_precision", ascending=False)
        winner = max(fitted, key=lambda name: fitted[name].best_score_)
        model = fitted[winner].best_estimator_
        # Threshold selection uses training data only; the final test is not consulted.
        oof_probability = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
        threshold_rows = [{"threshold": float(t), **metrics(y_train, oof_probability, t)} for t in np.linspace(0.10, 0.80, 71)]
        threshold_frame = pd.DataFrame(threshold_rows)
        threshold = float(threshold_frame.sort_values(["f2", "precision", "threshold"], ascending=False).iloc[0].threshold)
        oof_metrics = metrics(y_train, oof_probability, threshold)
        # GridSearchCV has already refit the selected pipeline on the complete training split.
        test_probability = model.predict_proba(X_test)[:, 1]
        test_metrics = metrics(y_test, test_probability, threshold)
        default_metrics = metrics(y_test, test_probability, 0.5)
        predicted = (test_probability >= threshold).astype(int)
        dummy_probability = np.full(len(y_test), y_train.mean())
        dummy_metrics = metrics(y_test, dummy_probability, 0.5)
        baseline_f2 = float(fbeta_score(y_test, np.ones(len(y_test)), beta=2))
        quality_gate = bool(test_metrics["average_precision"] > y_test.mean() and test_metrics["f2"] > baseline_f2)
        # Interval estimation resamples whole profile groups rather than correlated rows.
        rng = np.random.default_rng(SEED)
        group_indexes = {g: np.flatnonzero(g_test.to_numpy() == g) for g in g_test.unique()}
        unique_groups = np.array(list(group_indexes))
        bootstrap = []
        for _ in range(400):
            indices = np.concatenate([group_indexes[g] for g in rng.choice(unique_groups, len(unique_groups), replace=True)])
            ys, ps = y_test.iloc[indices], test_probability[indices]
            if ys.nunique() == 2:
                bootstrap.append({"average_precision": average_precision_score(ys, ps),
                                  "recall": recall_score(ys, ps >= threshold, zero_division=0)})
        confidence = {name: [float(v) for v in np.quantile([row[name] for row in bootstrap], [0.025, 0.975])]
                      for name in ["average_precision", "recall"]}
        # Report budget-based lift as a descriptive ranking result, not a revenue forecast.
        budget = int(np.ceil(0.2 * len(y_test)))
        top_index = np.argsort(-test_probability, kind="stable")[:budget]
        top_precision = float(y_test.iloc[top_index].mean())
        lift = top_precision / float(y_test.mean())
        importance = permutation_importance(model, X_test, y_test, scoring="average_precision", n_repeats=5,
                                           random_state=SEED, n_jobs=1)
        importance_frame = pd.DataFrame({"feature": FEATURES, "mean_ap_decrease": importance.importances_mean,
                                         "std_ap_decrease": importance.importances_std}).sort_values("mean_ap_decrease", ascending=False)
        model_path = DEPLOYMENT / "model.joblib"
        joblib.dump(model, model_path, compress=3)
        reloaded = joblib.load(model_path)
        max_delta = float(np.abs(reloaded.predict_proba(X_test)[:, 1] - test_probability).max())
        assert max_delta < 1e-12
        metadata = {
            "model": winner, "threshold": threshold, "features": FEATURES, "derived_features": DERIVED,
            "best_parameters": fitted[winner].best_params_, "cv_average_precision": float(fitted[winner].best_score_),
            "oof_metrics": oof_metrics, "test_metrics": test_metrics, "test_at_default_threshold": default_metrics,
            "dummy_test_metrics": dummy_metrics, "all_contact_baseline_f2": baseline_f2,
            "cluster_bootstrap_95pct": confidence, "top_20pct_precision": top_precision, "top_20pct_lift": lift,
            "top_20pct_count": budget, "source_sha256": manifest["source_sha256"], "model_sha256": sha256(model_path),
            "test_prevalence": float(y_test.mean()), "train_rows": len(X_train), "test_rows": len(X_test),
            "python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__,
            "numpy": np.__version__, "mlflow_parent_run_id": parent.info.run_id,
            "tuned_combinations": len(all_trials), "quality_gate_passed": quality_gate,
            "serialization_max_probability_difference": max_delta,
            "source_commit": os.environ.get("GITHUB_SHA", "local-execution"),
            "duration_seconds": round(time.monotonic() - started, 2),
        }
        write_json(DEPLOYMENT / "model_metadata.json", metadata)
        write_json(REPORTS / "metrics.json", metadata)
        pd.DataFrame(all_trials).to_csv(REPORTS / "cv_trials.csv", index=False)
        comparison_frame.to_csv(REPORTS / "model_comparison.csv", index=False)
        threshold_frame.to_csv(REPORTS / "threshold_search.csv", index=False)
        importance_frame.to_csv(REPORTS / "permutation_importance.csv", index=False)
        pd.DataFrame({"actual": y_test, "score": test_probability, "predicted": predicted}).to_csv(REPORTS / "test_predictions.csv", index=False)
        pd.DataFrame(classification_report(y_test, predicted, output_dict=True, zero_division=0)).T.to_csv(REPORTS / "classification_report.csv")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        ConfusionMatrixDisplay.from_predictions(y_test, predicted, display_labels=["No purchase", "Purchase"], ax=axes[0], colorbar=False, cmap="Blues")
        PrecisionRecallDisplay.from_predictions(y_test, test_probability, ax=axes[1], name=winner)
        axes[1].axhline(y_test.mean(), linestyle="--", color="gray", label="Prevalence")
        axes[1].legend(fontsize=8)
        RocCurveDisplay.from_predictions(y_test, test_probability, ax=axes[2], name=winner)
        axes[2].plot([0, 1], [0, 1], "--", color="gray")
        axes[0].set_title(f"Test predictions at threshold {threshold:.2f}")
        axes[1].set_title("Precision-recall curve")
        axes[2].set_title("ROC curve")
        fig.tight_layout()
        fig.savefig(REPORTS / "test_performance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_params({"winner": winner, "threshold": threshold, **fitted[winner].best_params_})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metrics({f"oof_{k}": v for k, v in oof_metrics.items()})
        mlflow.log_metric("top_20pct_lift", lift)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifacts(str(REPORTS), artifact_path="reports")
        print(json.dumps(metadata, indent=2), flush=True)
    # Record a promotion gate failure as an actual failed job, never as a success message.
    if not quality_gate:
        raise RuntimeError("Candidate failed the documented holdout quality gate; model promotion is blocked.")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=DATA / "splits")
    train(parser.parse_args().splits)
