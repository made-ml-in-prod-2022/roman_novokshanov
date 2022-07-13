from dataclasses import dataclass

from omegaconf import DictConfig
from .split_params import SchemaTrainParams
from .feature_params import FeaturesParams
from .train_params import ModelParams
from marshmallow_dataclass import class_schema


@dataclass()
class TrainingPipelineParams:
    model: ModelParams
    schema: SchemaTrainParams
    features: FeaturesParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(config: DictConfig) -> TrainingPipelineParams:
    schema = TrainingPipelineParamsSchema()
    return schema.load(config)
