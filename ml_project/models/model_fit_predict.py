import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline

from ml_project.enities.train_params import TrainingParams
from ml_project.features.build_features import FeaturesTransformer

SklearnRegressionModel = Union[
    LogisticRegression,
    RandomForestClassifier,
    KNeighborsClassifier,
]


def train_model(
    features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> SklearnRegressionModel:

    if train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(
            criterion="gini",
            n_estimators=250,
            random_state=train_params.random_state,
        )

    elif train_params.model_type == "KNeighborsClassifier":
        model = KNeighborsClassifier(
            n_neighbors=10,
        )

    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(
            C=1.0,
            solver="liblinear",
            random_state=train_params.random_state,
        )

    else:
        raise NotImplementedError()

    model.fit(features, target)
    return model


def predict_model(
    model: Pipeline, features: pd.DataFrame, use_log_trick: bool = False
) -> np.ndarray:
    predicts = model.predict(features)
    if use_log_trick:
        predicts = np.exp(predicts)
    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series, use_log_trick: bool = False
) -> Dict[str, float]:

    if use_log_trick:
        target = np.exp(target)

    return {
        "accuracy": accuracy_score(target, predicts),
        "r2_score": r2_score(target, predicts),
        "rmse": mean_squared_error(target, predicts, squared=False),
        "mae": mean_absolute_error(target, predicts),
    }


def create_inference_pipeline(
    model: SklearnRegressionModel, transformer: FeaturesTransformer
) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def serialize_model(model: SklearnRegressionModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output

def deserialize_model(path: str) -> SklearnRegressionModel:
    with open(path, "rb") as f:
        return pickle.load(f)
