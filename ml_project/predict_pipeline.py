#!/usr/bin/env python
import sys

import logging
import logging.config

import yaml
from omegaconf import DictConfig
import hydra

import pandas as pd

from ml_project.data.make_dataset import read_data
from ml_project.enities.predict_pipeline_params import read_predict_pipeline_params

from ml_project.models.model_fit_predict import (
    predict_model,
    deserialize_model,
)

DEFAULT_DATASET_PATH = "./data/raw/test.csv"
DEFAULT_CONFIG_PATH = "./configs/predict_config.yaml"
DEFAULT_LOGGING_CONFIG_FILEPATH = "./configs/predict_logging_config.yaml"

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def prepare_val_features_for_predict(
    train_features: pd.DataFrame, val_features: pd.DataFrame
):
    # small hack to work with categories
    train_features, val_features = train_features.align(
        val_features, join="left", axis=1
    )
    val_features = val_features.fillna(0)
    return val_features


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


def setup_logging():
    """ Setting up logging configuration """
    with open(
        hydra.utils.to_absolute_path(".") + "/" + DEFAULT_LOGGING_CONFIG_FILEPATH
    ) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


@hydra.main(config_path="../configs", config_name="predict_config")
def predict_pipeline_command(config: DictConfig) -> None:
    # setup_logging()
    predict_pipeline(config)


if __name__ == "__main__":
    predict_pipeline_command()
