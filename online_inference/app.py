import logging
import os
import sys
import pickle
from typing import List, Union, Optional

import numpy as np
import pandas as pd
import uvicorn
from omegaconf import DictConfig
import hydra
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from sklearn.pipeline import Pipeline

from src.make_dataset import read_data
from src.predict_pipeline_params import read_predict_pipeline_params

from src.model_fit_predict import (
    predict_model,
    deserialize_model,
)

DEFAULT_DATASET_PATH = "./data/test.csv"
DEFAULT_CONFIG_PATH = "./configs/predict_config.yaml"
DEFAULT_MODEL_PATH = "./models/model.pkl"


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


class HousePricesModel(BaseModel):
    data: List[conlist(Union[float, str, None], min_items=80, max_items=80)]
    features: List[str]


class PriceResponse(BaseModel):
    id: str
    price: float


model: Optional[Pipeline] = None


def predict_pipeline(config: DictConfig):

    # reading config, loading dataset
    logger.info(f"Reading predict pipeline params")
    predict_pipeline_params = read_predict_pipeline_params(config)

    return run_predict_pipeline(predict_pipeline_params)


def run_predict_pipeline(predict_params):
    """
    Process predict pipeline
    """

    current_path = hydra.utils.to_absolute_path(".") + "/"

    logger.info(f"Start predict pipeline with params: {predict_params}")

    df = read_data(current_path + predict_params.schema.input_data_path)
    logger.info(f"Data.shape is {df.shape}")

    df = df.drop(columns=predict_params.features.feature_params.target_col, axis=1)

    # load trained model
    model = deserialize_model(current_path + predict_params.schema.input_model_path)

    # get predictions
    prediction = predict_model(model, df)

    pd.Series(prediction, index=df.index, name="prediction").to_csv(
        current_path + predict_params.schema.output_predict_path
    )

    logger.info(f"Prediction is completed")


def predict_model(
    model: Pipeline, features: pd.DataFrame, use_log_trick: bool = False
) -> np.ndarray:
    predicts = model.predict(features)
    if use_log_trick:
        predicts = np.exp(predicts)
    return predicts


def make_predict(
    data: List, features: List[str], model: Pipeline,
) -> List[PriceResponse]:
    data = pd.DataFrame(data, columns=features)
    ids = [int(x) for x in data["Id"]]
    predicts = np.exp(model.predict(data))

    return [
        PriceResponse(id=id_, price=float(price)) for id_, price in zip(ids, predicts)
    ]


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@hydra.main(config_path="configs", config_name="predict_config")
@app.on_event("startup")
def predict_pipeline_command(config: DictConfig) -> None:
    logger.info(f"Startup config {config}...")
    # setup_logging()
    predict_pipeline(config)


def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)


@app.get("/health")
def health() -> bool:
    return not (model is None)


@app.get("/predict/", response_model=List[PriceResponse])
def predict(request: HousePricesModel):
    return make_predict(request.data, request.features, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
