import os
import pytest
from textwrap import dedent
import pandas as pd
import numpy as np
import yaml

from ml_project.train_pipeline import (
    train_pipeline,
    DEFAULT_DATASET_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_LOGGING_CONFIG_FILEPATH,
)

from ml_project.enities.train_pipeline_params import (
    read_training_pipeline_params,
)


TRAINING_PARAMS_STR = dedent(
    """\
    'model': 
        'model_params': 
            'model_type': 'RandomForestClassifier'
            'random_state': 42
    'schema': 
        'input_data_path': './ml_project/data/raw/train.csv'
        'output_model_path': './ml_project/models/model.pkl'
        'metric_path': './ml_project/models/metrics.json'
        'downloading_params':
            'use_download': 'True'
            's3_bucket': 'made-mlprod-hw1'
            'endpointurl': 'https://ib.bizmrg.com'
            'paths': 
                - 'train.csv'
                - 'test.csv'
            'output_folder': './data/raw/'
        'splitting_params':
            'val_size': 0.1
            'random_state': 42
        'mlflow':
            'use_mlflow': 'False'
            'mlflow_uri': 'http://localhost:5001'
            'mlflow_experiment': 'test'
    'features': 
        'feature_params': 
            'categorical_features': 
                - 'sex'
                - 'cp'
                - 'fbs'
                - 'restecg'
                - 'exang'
                - 'slope'
                - 'thal'
                - 'ca'
            'numerical_features': 
                - 'age'
                - 'trestbps'
                - 'chol'
                - 'thalach'
                - 'oldpeak'
            'target_col': 'condition'
            'use_log_trick': 'False'
    """
)


@pytest.fixture()
def training_params_fio(tmpdir):
    fio = tmpdir.join("training_params.yaml")
    fio.write(TRAINING_PARAMS_STR, "w")
    return fio


def test_can_read_training_pipeline_params(training_params_fio, caplog):
    caplog.set_level("DEBUG")
    with caplog.at_level("DEBUG"):

        with open(training_params_fio, "r") as input_stream:
            training_params_config = yaml.safe_load(input_stream)

        training_params_loaded = read_training_pipeline_params(
            training_params_config)

        training_params_local = {
            "input_data_path": "./ml_project/data/raw/train.csv",
            "output_model_path": "./ml_project/models/model.pkl",
            "train_params": {
                "model_type": "RandomForestClassifier",
                "random_state": 42,
            },
            "feature_params": {
                "categorical_features": [
                    "sex",
                    "cp",
                    "fbs",
                    "restecg",
                    "exang",
                    "slope",
                    "thal",
                    "ca",
                ],
            },
            "target_col": "condition",
        }

        assert (
            training_params_local["input_data_path"]
            == training_params_loaded.schema.input_data_path
        )

        assert (
            training_params_local["feature_params"]["categorical_features"]
            == training_params_loaded.features.feature_params.categorical_features
        )

        assert (
            training_params_local["target_col"]
            == training_params_loaded.features.feature_params.target_col
        )


# testing build_features functionality


def generate_random_dataset(random_state=42, size=100):
    np.random.seed(random_state)
    data = pd.DataFrame()

    data["age"] = np.random.uniform(low=5.0, high=100.0, size=size)
    data["sex"] = np.random.choice(
        [0, 1], replace=True, p=[0.5] * 2, size=size)
    data["cp"] = np.random.choice(
        [0, 1, 2, 3], replace=True, p=[0.25] * 4, size=size)
    data["trestbps"] = np.random.uniform(low=50.0, high=250.0, size=size)
    data["chol"] = np.random.uniform(low=50.0, high=400.0, size=size)
    data["fbs"] = np.random.choice(
        [0, 1], replace=True, p=[0.5] * 2, size=size)
    data["restecg"] = np.random.choice(
        [0, 1, 2], replace=True, p=[1.0 / 3.0] * 3, size=size
    )
    data["thalach"] = np.random.uniform(low=70.0, high=200.0, size=size)
    data["exang"] = np.random.choice(
        [0, 1], replace=True, p=[0.5] * 2, size=size)
    data["oldpeak"] = np.random.uniform(low=0.0, high=7.0, size=size)
    data["slope"] = np.random.choice(
        [0, 1, 2], replace=True, p=[1.0 / 3] * 3, size=size
    )
    data["ca"] = np.random.choice(
        [0, 1, 2, 3, 4], replace=True, p=[0.2] * 5, size=size)
    data["thal"] = np.random.choice(
        [0, 1, 2, 3], replace=True, p=[0.25] * 4, size=size)
    data["condition"] = np.random.choice(
        [0, 1], replace=True, p=[0.5] * 2, size=size)

    return data


@pytest.fixture()
def dataset_fio(tmpdir):
    data = generate_random_dataset(random_state=42, size=100)
    fio = tmpdir.join("data.csv")
    data.to_csv(fio)
    return fio


@pytest.fixture()
def training_params_modified_fio(dataset_fio, tmpdir):

    config_str = dedent(
        """\
            'model': 
                'model_params': 
                    'model_type': 'RandomForestClassifier'
                    'random_state': 42
            'schema': 
                'input_data_path': {}
                'output_model_path': None
                'metric_path': None
                'downloading_params':
                    'use_download': 'True'
                    's3_bucket': 'made-mlprod-hw1'
                    'endpointurl': 'https://ib.bizmrg.com'
                    'paths': 
                        - 'train.csv'
                        - 'test.csv'
                    'output_folder': './data/raw/'
                'splitting_params':
                    'val_size': 0.1
                    'random_state': 42
                'mlflow':
                    'use_mlflow': 'False'
                    'mlflow_uri': 'http://localhost:5001'
                    'mlflow_experiment': 'test' 
            'features': 
                'feature_params': 
                    'categorical_features': 
                        - 'sex'
                        - 'cp'
                        - 'fbs'
                        - 'restecg'
                        - 'exang'
                        - 'slope'
                        - 'thal'
                        - 'ca'
                    'numerical_features': 
                        - 'age'
                        - 'trestbps'
                        - 'chol'
                        - 'thalach'
                        - 'oldpeak'
                    'target_col': 'condition'
                    'use_log_trick': 'False'
            """.format(os.path.relpath(str(dataset_fio)))
    )

    fio = tmpdir.join("training_params_modified.yaml")
    fio.write(config_str, "w")
    return fio


def test_train_pipeline(training_params_modified_fio, caplog):
    caplog.set_level("DEBUG")
    with caplog.at_level("DEBUG"):

        with open(training_params_modified_fio, "r") as input_stream:
            training_params_modified_config = yaml.safe_load(input_stream)

        train_pipeline(training_params_modified_config)

        assert (
            "Reading train pipeline params" in caplog.text
        ), "Train pipeline failed"

        assert (
            "Start train pipeline with params:" in caplog.text
        ), "Train pipeline failed"

        assert "Data.shape is" in caplog.text, "Train pipeline failed"

        assert "Train_df.shape is" in caplog.text, "Train pipeline failed"

        assert "Train_features.shape is" in caplog.text, "Train pipeline failed"

        assert "Val_features.shape is" in caplog.text, "Train pipeline failed"

        assert "Training is completed." in caplog.text, "Train pipeline failed"
