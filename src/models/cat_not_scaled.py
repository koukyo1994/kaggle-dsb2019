import numpy as np
import pandas as pd

from typing import Tuple, Union, Optional

from catboost import CatBoostClassifier, CatBoostRegressor

from .base import BaseModel
from src.evaluation import (CatBoostOptimizedNotScaled,
                            OptimizedRounderNotScaled)

CatModel = Union[CatBoostClassifier, CatBoostRegressor]


class CatBoostNotScaledModel(BaseModel):
    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_valid: np.ndarray, y_valid: np.ndarray,
            x_valid2: Optional[np.ndarray], y_valid2: Optional[np.ndarray],
            config: dict) -> Tuple[CatModel, dict]:
        model_params = config["model"]["model_params"]
        mode = config["model"]["mode"]
        self.mode = mode
        if mode == "regression":
            model = CatBoostRegressor(
                eval_metric=CatBoostOptimizedNotScaled(), **model_params)
        else:
            model = CatBoostClassifier(**model_params)

        if x_valid2 is not None:
            eval_sets = [(x_valid2, y_valid2)]
        else:
            eval_sets = [(x_valid, y_valid)]

        model.fit(
            x_train,
            y_train,
            eval_set=eval_sets,
            use_best_model=True,
            verbose=model_params["early_stopping_rounds"])
        best_score = model.best_score_
        return model, best_score

    def get_best_iteration(self, model: CatModel):
        return model.best_iteration_

    def predict(self, model: CatModel,
                features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return model.predict(features)

    def get_feature_importance(self, model: CatModel) -> np.ndarray:
        return model.feature_importances_

    def post_process(self, oof_preds: np.ndarray, test_preds: np.ndarray,
                     valid_preds: Optional[np.ndarray], y: np.ndarray,
                     y_valid: Optional[np.ndarray], config: dict
                     ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        # Override
        if self.mode == "regression":
            OptR = OptimizedRounderNotScaled()
            OptR.fit(oof_preds, y)
            coef = OptR.coefficients()
            oof_preds_ = OptR.predict(oof_preds, coef)
            test_preds_ = OptR.predict(test_preds, coef)
            if valid_preds is not None:
                valid_preds = OptR.predict(valid_preds, coef)
            return oof_preds_, test_preds_, valid_preds
        return oof_preds, test_preds, valid_preds
