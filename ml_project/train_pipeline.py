#!/usr/bin/env python
import os
import sys

import logging
import logging.config
import json
from pathlib import Path
from urllib.parse import urlparse

import yaml
from omegaconf import DictConfig
import hydra

import pandas as pd
import mlflow
import mlflow.sklearn

from ml_project.data.make_dataset import read_data, split_train_val_data
from ml_project.data.make_dataset import download_data_from_s3
from ml_project.enities.train_pipeline_params import read_training_pipeline_params

from ml_project.features.build_features import extract_target
from ml_project.models.model_fit_predict import (
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)

from ml_project.features.build_features import make_features
from ml_project.features.build_features import build_transformer
from ml_project.models.model_fit_predict import create_inference_pipeline

DEFAULT_DATASET_PATH = "./data/raw/train.csv"
DEFAULT_CONFIG_PATH = "./ml_project/configs/train_config.yaml"
DEFAULT_LOGGING_CONFIG_FILEPATH = "./ml_project/configs/train_logging_config.yaml"

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def train_pipeline(config: DictConfig):

    # reading config, loading dataset
    logger.info(f"Reading train pipeline params")    
    training_pipeline_params = read_training_pipeline_params(config)
    
    logger.info(f"mlflow used: {training_pipeline_params.schema.mlflow.use_mlflow}")
    if training_pipeline_params.schema.mlflow.use_mlflow:

        mlflow.set_tracking_uri(training_pipeline_params.schema.mlflow.mlflow_uri)
        mlflow.set_experiment(training_pipeline_params.schema.mlflow.mlflow_experiment)
        with mlflow.start_run():
            model_path, metrics, model = run_train_pipeline(training_pipeline_params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(os.getcwd() + '/.hydra')

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                mlflow.sklearn.log_model(model, "model", registered_model_name="Model " + training_pipeline_params.model.model_params.model_type)
            else:
                mlflow.sklearn.log_model(model, "model")

    else:
        return run_train_pipeline(training_pipeline_params)

def run_train_pipeline(train_params):
    """
    Process train pipeline
    """

    current_path = hydra.utils.to_absolute_path(".") + "/"

    # downloading data
    downloading_params = train_params.schema.downloading_params

    if downloading_params.use_download:
        logger.info(f"Downloading data with params: {downloading_params}")
        os.makedirs(downloading_params.output_folder, exist_ok=True)
        for path in downloading_params.paths:
            download_data_from_s3(
                downloading_params.s3_bucket,
                path,
                os.path.join(current_path, downloading_params.output_folder, Path(path).name),
                endpoint_url=downloading_params.endpointurl,
            )

    logger.info(f"Start train pipeline with params: {train_params}")

    data = read_data(current_path + train_params.schema.input_data_path)
    logger.info(f"Data.shape is {data.shape}")
    train_df, val_df = split_train_val_data(
        data, train_params.schema.splitting_params)

    logger.info(f"Train_df.shape is {train_df.shape}")
    logger.info(f"Val_df.shape is {val_df.shape}")

    val_target = extract_target(val_df, train_params.features.feature_params)
    train_target = extract_target(train_df, train_params.features.feature_params)
    train_df = train_df.drop(columns=train_params.features.feature_params.target_col, axis=1)
    val_df = val_df.drop(columns=train_params.features.feature_params.target_col, axis=1)

    # build features
    transformer = build_transformer(train_params.features.feature_params)
    transformer.fit(train_df)
    train_features = make_features(transformer, train_df)
    logger.info(f"Train_features.shape is {train_features.shape}")
    logger.info(f"Val_features.shape is {val_df.shape}")

    # train model
    model = train_model(train_features, train_target,
                        train_params.model.model_params)
    
    # get predictions
    inference_pipeline = create_inference_pipeline(model, transformer)
    predicts = predict_model(inference_pipeline, val_df,
        use_log_trick=train_params.features.feature_params.use_log_trick)

    # evaluate prediction
    metrics = evaluate_model(predicts, val_target,
        use_log_trick=train_params.features.feature_params.use_log_trick)

    if train_params.schema.metric_path is not None:
        with open(current_path + train_params.schema.metric_path, "w") as metric_file:
            json.dump(metrics, metric_file)
        logger.info(f"Metrics is {metrics}")

    if train_params.schema.output_model_path is not None:
        path_to_model = serialize_model(
            inference_pipeline, current_path + train_params.schema.output_model_path
        )
        logger.info(f"Model is serialized to {path_to_model}")

    logger.info(f"Training is completed.")
    
    return path_to_model, metrics, model


def setup_logging():
    """ Setting up logging configuration """
    with open(
        hydra.utils.to_absolute_path(
            ".") + "/" + DEFAULT_LOGGING_CONFIG_FILEPATH
    ) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


@hydra.main(config_path="configs", config_name="train_config")
def train_pipeline_command(config: DictConfig) -> None:
    #setup_logging()
    train_pipeline(config)


if __name__ == "__main__":
    train_pipeline_command()
