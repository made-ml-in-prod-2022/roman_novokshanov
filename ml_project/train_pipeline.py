#!/usr/bin/env python

import logging
import logging.config
import json

import yaml
from omegaconf import DictConfig
import hydra

import pandas as pd

from ml_project.data.make_dataset import read_data, split_train_val_data
from ml_project.enities.train_pipeline_params import read_training_pipeline_params

from ml_project.features.build_features import extract_target, FeaturesTransformer
from ml_project.models.model_fit_predict import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)

APPLICATION_NAME = "train_pipeline"
DEFAULT_DATASET_PATH = "./data/raw/heart.csv"
DEFAULT_CONFIG_PATH = "./ml_project/configs/train_config.yaml"
DEFAULT_LOGGING_CONFIG_FILEPATH = "./ml_project/configs/train_logging_config.yaml"

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


def train_pipeline(config):
    """
    Process train pipeline
    """
    current_path = hydra.utils.to_absolute_path(".") + "/"

    # reading config, loading dataset
    logger.info(f"Reading train pipeline params")
    train_params = read_training_pipeline_params(config)

    logger.info(f"Start train pipeline with params: {train_params}")

    data = read_data(current_path + train_params.schema.input_data_path)
    logger.info(f"Data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(
        data, train_params.schema.splitting_params)
    logger.info(f"Train_df.shape is {train_df.shape}")
    logger.info(f"Val_df.shape is {val_df.shape}")

    # build features
    features = FeaturesTransformer(train_params.features.feature_params)
    features.fit(train_df)

    train_features = features.transform(train_df)
    train_target = extract_target(
        train_df, train_params.features.feature_params)
    logger.info(f"Train_features.shape is {train_features.shape}")

    val_features = features.transform(val_df)
    val_target = extract_target(val_df, train_params.features.feature_params)
    val_features_prepared = prepare_val_features_for_predict(
        train_features, val_features
    )
    logger.info(f"Val_features.shape is {val_features_prepared.shape}")

    # train model
    model = train_model(train_features, train_target,
                        train_params.model.model_params)

    # get predictions
    predicts = predict_model(model, val_features_prepared)

    # evaluate prediction
    metrics = evaluate_model(predicts, val_target)

    if train_params.schema.metric_path is not None:
        with open(current_path + train_params.schema.metric_path, "w") as metric_file:
            json.dump(metrics, metric_file)
        logger.info(f"Metrics is {metrics}")

    if train_params.schema.output_model_path is not None:
        path_to_model = serialize_model(
            model, current_path + train_params.schema.output_model_path
        )
        logger.info(f"Model is serialized to {path_to_model}")
    logger.info(f"Training is completed.")
    return path_to_model, metrics


def setup_logging():
    """ Setting up logging configuration """
    with open(
        hydra.utils.to_absolute_path(
            ".") + "/" + DEFAULT_LOGGING_CONFIG_FILEPATH
    ) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


@hydra.main(config_path="configs", config_name="train_config")
def train_pipeline_command(config: DictConfig) -> None:
    setup_logging()
    train_pipeline(config)


if __name__ == "__main__":
    train_pipeline_command()
