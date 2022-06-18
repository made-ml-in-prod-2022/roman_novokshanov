# ml_project

## homework 1 for MADE course on ML in production 2021

## Setting up environment

Requirements: `conda, python >= 3.7`

Installation:

**conda:**

`conda create --name myenv python=3.7 --file requirements.txt -c conda-forge`

`conda activate myenv`

**pip:**

`pip install -r requirements.txt`

## Data download

`dvc pull`

## Tests

`python -m pytest tests -v --cov --cov-fail-under=80`

## EDA report

Jupyter notebook:
`ml_project/notebooks/1_0_nr_initial_data_exploration.ipynb`

## Training

* default config (random forest):

  `python -m ml_project.train_pipeline`
* default config with model specified (rf, lr):

  `python -m ml_project.train_pipeline model=lr`

## Prediction

* default config (random forest):

`python -m ml_project.predict_pipeline`

* default config with model specified (rf, lr):

`python -m ml_project.predict_pipeline model=lr`

## Mlflow server UI:

`mlflow server --backend-store-uri sqlite:///mydb.sqlite --default-artifact-root $(pwd)/ml_project/mlruns/ -p 5001`


## Run in Docker container
python setup.py sdist
docker build -t ml_project_train:v1 ./
docker run --rm -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} ml_project_train:v1

mount local folder:
docker run --rm -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} -v ${PWD}/models:/models -v ${PWD}/outputs:/outputs ml_project_train:v1