"""Тренує модель з MLflow tracking + cross-validation.

Запуск:
    python -m src.train                          # усі 7 експериментів
    python -m src.train --only svc               # лише SVC (3 експерименти)
    python -m src.train --experiment custom-name # власна назва експерименту
"""
import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from src.pipeline import createPipeline

DATA_PATH = Path("data/raw/wine.csv")
MODEL_DIR = Path("models")
EXPERIMENT_NAME = "wine-classification"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


@dataclass
class ExperimentConfig:
    runName: str
    modelName: str
    params: dict[str, Any]


EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig("svc_rbf_C1", "svc", {"C": 1.0, "kernel": "rbf", "gamma": "scale"}),
    ExperimentConfig("svc_rbf_C10", "svc", {"C": 10.0, "kernel": "rbf", "gamma": "scale"}),
    ExperimentConfig("svc_linear_C1", "svc", {"C": 1.0, "kernel": "linear"}),
    ExperimentConfig("logreg_l2_C1", "logreg", {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}),
    ExperimentConfig("logreg_l2_C01", "logreg", {"C": 0.1, "penalty": "l2", "solver": "lbfgs"}),
    ExperimentConfig("gb_n100_lr01", "gradient_boosting", {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}),
    ExperimentConfig("gb_n200_lr005", "gradient_boosting", {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3}),
]


def loadData(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python -m src.create_dataset (or `dvc pull`)."
        )
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def computeMetrics(yTrue: np.ndarray, yPred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(yTrue, yPred)),
        "precision": float(precision_score(yTrue, yPred, average="weighted")),
        "recall": float(recall_score(yTrue, yPred, average="weighted")),
        "f1": float(f1_score(yTrue, yPred, average="weighted")),
    }


def trainOne(config: ExperimentConfig, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = createPipeline(config.modelName, **config.params)

    cvScores = cross_val_score(
        pipeline, XTrain, yTrain, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1
    )

    pipeline.fit(XTrain, yTrain)
    yPredTrain = pipeline.predict(XTrain)
    yPredTest = pipeline.predict(XTest)

    trainMetrics = computeMetrics(yTrain, yPredTrain)
    testMetrics = computeMetrics(yTest, yPredTest)

    with mlflow.start_run(run_name=config.runName) as run:
        mlflow.log_param("model_name", config.modelName)
        mlflow.log_param("dataset", "wine")
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("cv_folds", CV_FOLDS)
        for key, value in config.params.items():
            mlflow.log_param(key, value)

        mlflow.log_metric("cv_mean_accuracy", float(cvScores.mean()))
        mlflow.log_metric("cv_std_accuracy", float(cvScores.std()))
        for key, value in trainMetrics.items():
            mlflow.log_metric(f"train_{key}", value)
        for key, value in testMetrics.items():
            mlflow.log_metric(f"test_{key}", value)

        mlflow.set_tag("model_type", config.modelName)
        mlflow.set_tag("variant", "2")

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        return {
            "run_id": run.info.run_id,
            "run_name": config.runName,
            "model_name": config.modelName,
            "params": config.params,
            "cv_mean_accuracy": float(cvScores.mean()),
            "cv_std_accuracy": float(cvScores.std()),
            "train_metrics": trainMetrics,
            "test_metrics": testMetrics,
            "pipeline": pipeline,
        }


def selectBest(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(results, key=lambda r: r["test_metrics"]["accuracy"])


def saveBestPipeline(best: dict[str, Any], outputPath: Path) -> None:
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    with open(outputPath, "wb") as f:
        pickle.dump(best["pipeline"], f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Wine classifiers with MLflow tracking")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME, help="MLflow experiment name")
    parser.add_argument("--only", choices=["svc", "logreg", "gradient_boosting"], help="Run only one model family")
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment)

    X, y = loadData()
    print(f"Loaded dataset: X={X.shape}, y={y.shape}, classes={sorted(y.unique().tolist())}")

    configs = EXPERIMENTS if args.only is None else [c for c in EXPERIMENTS if c.modelName == args.only]

    results = []
    for config in configs:
        print(f"\n[run] {config.runName} ({config.modelName})")
        result = trainOne(config, X, y)
        results.append(result)
        print(
            f"  cv={result['cv_mean_accuracy']:.4f}±{result['cv_std_accuracy']:.4f}  "
            f"test_acc={result['test_metrics']['accuracy']:.4f}  "
            f"test_f1={result['test_metrics']['f1']:.4f}"
        )

    if not results:
        print("No experiments matched the filter.")
        return

    best = selectBest(results)
    saveBestPipeline(best, MODEL_DIR / "best_pipeline.pkl")

    summary = {
        "best_run_id": best["run_id"],
        "best_run_name": best["run_name"],
        "best_test_accuracy": best["test_metrics"]["accuracy"],
        "saved_to": str(MODEL_DIR / "best_pipeline.pkl"),
        "all_results": [
            {
                "run_name": r["run_name"],
                "run_id": r["run_id"],
                "cv_mean_accuracy": r["cv_mean_accuracy"],
                "test_accuracy": r["test_metrics"]["accuracy"],
                "test_f1": r["test_metrics"]["f1"],
            }
            for r in results
        ],
    }
    (MODEL_DIR / "best_run.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Summary ===")
    print(f"Best run: {best['run_name']} (run_id={best['run_id']})")
    print(f"Best test accuracy: {best['test_metrics']['accuracy']:.4f}")
    print(f"Best pipeline saved: {MODEL_DIR / 'best_pipeline.pkl'}")


if __name__ == "__main__":
    main()
