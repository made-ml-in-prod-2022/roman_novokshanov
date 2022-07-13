#!/usr/bin/env python

import logging
import logging.config

import yaml
from omegaconf import DictConfig
import hydra

import pandas as pd

from ml_project.data.make_dataset import read_data, split_train_val_data
from ml_project.enities.predict_pipeline_params import read_predict_pipeline_params

from ml_project.features.build_features import extract_target, FeaturesTransformer
from ml_project.models.model_fit_predict import (
    predict_model,
    deserialize_model,
)

APPLICATION_NAME = "predict_pipeline"
DEFAULT_DATASET_PATH = "./data/raw/heart.csv"
DEFAULT_CONFIG_PATH = "./ml_project/configs/predict_config.yaml"
DEFAULT_LOGGING_CONFIG_FILEPATH = "./ml_project/configs/predict_logging_config.yaml"

logger = logging.getLogger(APPLICATION_NAME)


def prepare_val_features_for_predict(
    train_features: pd.DataFrame, val_features: pd.DataFrame
):
    # small hack to work with categories
    train_features, val_features = train_features.align(
        val_features, join="left", axis=1
    )
    val_features = val_features.fillna(0)
    return val_features


def predict_pipeline(config):
    """
    Process predict pipeline
    """

    current_path = hydra.utils.to_absolute_path(".") + "/"

    # reading config, loading dataset
    logger.info(f"Reading predict pipeline params")

    predict_params = read_predict_pipeline_params(config)
    logger.info(f"Start predict pipeline with params: {predict_params}")
    # data = read_data(dataset_filepath)
    data = read_data(current_path + predict_params.schema.input_data_path)
    logger.info(f"Data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(
        data, predict_params.schema.splitting_params
    )
    logger.info(f"Train_df.shape is {train_df.shape}")
    logger.info(f"Val_df.shape is {val_df.shape}")

    # build features
    features = FeaturesTransformer(predict_params.features.feature_params)
    features.fit(train_df)

    train_features = features.transform(train_df)
    train_target = extract_target(
        train_df, predict_params.features.feature_params)

    val_features = features.transform(val_df)
    val_target = extract_target(val_df, predict_params.features.feature_params)
    val_features_prepared = prepare_val_features_for_predict(
        train_features, val_features
    )
    logger.info(f"Val_features.shape is {val_features_prepared.shape}")

    # load trained model
    model = deserialize_model(
        current_path + predict_params.schema.input_model_path)

    # get predictions
    prediction = predict_model(model, val_features_prepared)

    pd.Series(prediction, index=val_features_prepared.index, name="prediction").to_csv(
        current_path + predict_params.schema.output_predict_path
    )

    logger.info(f"Prediction is completed")


def setup_logging():
    """ Setting up logging configuration """
    with open(
        hydra.utils.to_absolute_path(
            ".") + "/" + DEFAULT_LOGGING_CONFIG_FILEPATH
    ) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


@hydra.main(config_path="configs", config_name="predict_config")
def predict_pipeline_command(config: DictConfig) -> None:
    setup_logging()
    predict_pipeline(config)


if __name__ == "__main__":
    predict_pipeline_command()
