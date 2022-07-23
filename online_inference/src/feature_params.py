from dataclasses import dataclass
from typing import List, Optional
from xmlrpc.client import Boolean


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    target_col: Optional[str]
    use_log_trick: Optional[Boolean]


@dataclass()
class FeaturesParams:
    feature_params: FeatureParams
