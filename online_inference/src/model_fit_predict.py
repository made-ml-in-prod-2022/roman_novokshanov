import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def predict_model(
    model: Pipeline, features: pd.DataFrame, use_log_trick: bool = False
) -> np.ndarray:
    predicts = model.predict(features)
    if use_log_trick:
        predicts = np.exp(predicts)
    return predicts


def deserialize_model(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)
