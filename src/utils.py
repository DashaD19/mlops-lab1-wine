"""Допоміжні функції завантаження та використання моделі."""
import pickle
from pathlib import Path
from typing import Any

import mlflow.sklearn
import numpy as np
import pandas as pd


def loadModelFromFile(path: Path | str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def loadModelFromMlflow(runId: str, artifactPath: str = "model") -> Any:
    modelUri = f"runs:/{runId}/{artifactPath}"
    return mlflow.sklearn.load_model(modelUri)


def predict(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        X = X.values
    return model.predict(X)
