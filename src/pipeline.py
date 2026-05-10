"""Фабрика scikit-learn Pipeline для класифікації Wine.

Pipeline = StandardScaler -> classifier. Масштабування критичне для SVC та LogReg
(метричні моделі), Gradient Boosting нечутливий, але pipeline уніфікує API.
"""
from typing import Any

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODELS = {
    "svc": SVC,
    "logreg": LogisticRegression,
    "gradient_boosting": GradientBoostingClassifier,
}


def createPipeline(modelName: str, **modelParams: Any) -> Pipeline:
    if modelName not in MODELS:
        raise ValueError(
            f"Unknown model '{modelName}'. Available: {list(MODELS.keys())}"
        )

    modelParams.setdefault("random_state", 42)
    if modelName == "logreg":
        modelParams.setdefault("max_iter", 1000)

    classifier = MODELS[modelName](**modelParams)
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


if __name__ == "__main__":
    pipe = createPipeline("svc", C=1.0, kernel="rbf")
    print("Pipeline steps:", list(pipe.named_steps.keys()))
    print("Classifier:", pipe.named_steps["classifier"])
