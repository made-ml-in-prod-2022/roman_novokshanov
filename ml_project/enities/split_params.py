from dataclasses import dataclass, field
from typing import List

@dataclass()
class SplittingParams:

    val_size: float = field(default=0.2)
    random_state: int = field(default=42)

@dataclass()
class DownloadingParams:
    
    use_download: bool = field(default=True)
    s3_bucket: str = field(default='made-mlprod-hw1')
    endpointurl: str = field(default='https://ib.bizmrg.com')
    #paths: List = field(default_factory=lambda: ['train.csv', 'test.csv'])
    paths: List[str] = field(default_factory=lambda: ['train.csv', 'test.csv'])
    output_folder: str = field(default='./data/raw/')

@dataclass()
class MlflowParams:
    
    use_mlflow: bool = field(default=False)
    mlflow_uri: str = field(default='http://localhost:5001')
    mlflow_experiment: str = field(default='test')


@dataclass()
class SchemaTrainParams:
    input_data_path: str
    output_model_path: str
    metric_path: str    
    downloading_params: DownloadingParams
    splitting_params: SplittingParams
    mlflow: MlflowParams



@dataclass()
class SchemaPredictParams:
    input_data_path: str
    input_model_path: str
    output_predict_path: str
