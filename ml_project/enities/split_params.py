from dataclasses import dataclass, field


@dataclass()
class SplittingParams:

    val_size: float = field(default=0.2)
    random_state: int = field(default=42)


@dataclass()
class SchemaTrainParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams


@dataclass()
class SchemaPredictParams:
    input_data_path: str
    input_model_path: str
    output_predict_path: str
    splitting_params: SplittingParams
