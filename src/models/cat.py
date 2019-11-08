import numpy as np
import pandas as pd

from typing import Tuple, Union

from catboost import CatBoostClassifier, CatBoostRegressor
from easydict import EasyDict as edict

from .base import BaseModel

CatModel = Union[CatBoostClassifier, CatBoostRegressor]


class CatBoost(BaseModel):
    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_valid: np.ndarray, y_valid: np.ndarray,
            config: edict) -> Tuple[CatModel, dict]:
        model_params = config.model.model_params
        mode = config.model.train_params.mode
        if mode == "regression":
            model = CatBoostRegressor(**model_params)
        else:
            model = CatBoostClassifier(**model_params)

        model.fit(
            x_train,
            y_train,
            eval_set=(x_valid, y_valid),
            use_best_model=True,
            verbose=model_params.early_stopping_rounds)
        best_score = model.best_score_
        return model, best_score

    def get_best_iteration(self, model: CatModel):
        return model.best_iteration_

    def predict(self, model: CatModel,
                features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return model.predict(features)

    def get_feature_importance(self, model: CatModel) -> np.ndarray:
        return model.feature_importances_
