import lightgbm as lgb
import numpy as np
import pandas as pd

from typing import Union, Tuple

from .base import BaseModel

LGBMModel = Union[lgb.LGBMClassifier, lgb.LGBMRegressor]


class LightGBM(BaseModel):
    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_valid: np.ndarray, y_valid: np.ndarray,
            config: dict) -> Tuple[LGBMModel, dict]:
        d_train = lgb.Dataset(x_train, label=y_train)
        d_valid = lgb.Dataset(x_valid, label=y_valid)

        model_params = config["model"]["model_params"]
        train_params = config["model"]["train_params"]

        self.model_params = model_params
        self.train_params = train_params

        model = lgb.train(
            params=model_params,
            train_set=d_train,
            valid_sets=[d_valid],
            valid_names=["valid"],
            **train_params)
        best_score = dict(model.best_score)
        return model, best_score

    def get_best_iteration(self, model: LGBMModel) -> int:
        return model.best_iteration

    def predict(self, model: LGBMModel,
                features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return model.predict(features)

    def get_feature_importance(self, model: LGBMModel) -> np.ndarray:
        return model.feature_importance(importance_type="gain")
