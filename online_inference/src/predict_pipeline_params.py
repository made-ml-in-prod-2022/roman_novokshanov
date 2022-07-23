from dataclasses import dataclass
from .split_params import SchemaPredictParams
from .feature_params import FeaturesParams
from .predict_params import ModelParams
from marshmallow_dataclass import class_schema


@dataclass()
class PredictPipelineParams:
    model: ModelParams
    schema: SchemaPredictParams
    features: FeaturesParams


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_pipeline_params(config) -> PredictPipelineParams:
    schema = PredictPipelineParamsSchema()
    return schema.load(config)
