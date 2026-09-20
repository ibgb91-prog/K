"""Flask API serving a trained Iris flower classification model."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from flask import Flask, jsonify, request
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

FEATURE_NAMES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]


def train_model() -> tuple[Pipeline, list[str], float]:
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    classifier = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, random_state=42),
            ),
        ]
    )
    classifier.fit(x_train, y_train)
    test_accuracy = float(classifier.score(x_test, y_test))

    return classifier, iris.target_names.tolist(), test_accuracy


MODEL, CLASS_NAMES, TEST_ACCURACY = train_model()


def parse_features(payload: Any) -> np.ndarray:
    if not isinstance(payload, dict):
        raise ValueError("The request body must be a JSON object.")

    if "features" in payload:
        values = payload["features"]
        if not isinstance(values, list) or len(values) != len(FEATURE_NAMES):
            raise ValueError(
                f"'features' must be an array of exactly {len(FEATURE_NAMES)} numbers."
            )
    else:
        missing = [name for name in FEATURE_NAMES if name not in payload]
        if missing:
            raise ValueError(
                "Provide either a 'features' array or all named fields. "
                f"Missing: {', '.join(missing)}."
            )
        values = [payload[name] for name in FEATURE_NAMES]

    try:
        sample = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All feature values must be numeric.") from exc

    if sample.shape != (len(FEATURE_NAMES),):
        raise ValueError("Feature values must form a one-dimensional array.")
    if not np.all(np.isfinite(sample)):
        raise ValueError("Feature values must be finite numbers.")
    if np.any(sample < 0):
        raise ValueError("Flower measurements cannot be negative.")

    return sample.reshape(1, -1)


@app.get("/")
def index():
    return jsonify(
        {
            "service": "Iris flower classifier",
            "status": "ready",
            "endpoint": "POST /predict",
            "feature_order": FEATURE_NAMES,
            "example": {"features": [5.1, 3.5, 1.4, 0.2]},
        }
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "model": "LogisticRegression",
            "test_accuracy": round(TEST_ACCURACY, 4),
        }
    )


@app.post("/predict")
def predict():
    try:
        sample = parse_features(request.get_json(silent=True))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    predicted_index = int(MODEL.predict(sample)[0])
    probabilities = MODEL.predict_proba(sample)[0]

    return jsonify(
        {
            "prediction": CLASS_NAMES[predicted_index],
            "class_index": predicted_index,
            "confidence": round(float(probabilities[predicted_index]), 6),
            "probabilities": {
                class_name: round(float(probability), 6)
                for class_name, probability in zip(CLASS_NAMES, probabilities)
            },
            "input": {
                name: float(value)
                for name, value in zip(FEATURE_NAMES, sample[0])
            },
        }
    )


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Endpoint not found."}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
