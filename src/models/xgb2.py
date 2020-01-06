import numpy as np
import pandas as pd
import xgboost as xgb

from typing import Tuple, Union

from .base2 import BaseModel2
from src.evaluation import OptimizedRounder

XGBModel = Union[xgb.XGBClassifier, xgb.XGBRegressor]


class XGBoost2(BaseModel2):
    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_valid: np.ndarray, y_valid: np.ndarray,
            config: dict) -> Tuple[XGBModel, dict]:
        model_params = config["model"]["model_params"]
        train_params = config["model"]["train_params"]

        mode = config["model"]["mode"]
        self.mode = mode

        if mode == "regression":
            self.denominator = y_train.max()
            y_train = y_train / self.denominator
            y_valid = y_valid / self.denominator
            model = xgb.XGBRegressor(**model_params)
        else:
            model = xgb.XGBClassifier(**model_params)

        model.fit(
            x_train, y_train, eval_set=[(x_valid, y_valid)], **train_params)
        best_score = model.best_score
        return model, best_score

    def get_best_iteration(self, model: XGBModel):
        return model.best_iteration

    def predict(self, model: XGBModel,
                features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.mode != "multiclass":
            return model.predict(features, ntree_limit=model.best_ntree_limit)
        else:
            preds = model.predict_proba(
                features, ntree_limit=model.best_ntree_limit)
            return preds @ np.arange(4) / 3

    def get_feature_importance(self, model: XGBModel) -> np.ndarray:
        return model.feature_importances_

    def post_process(self, oof_preds: np.ndarray, test_preds: np.ndarray,
                     y: np.ndarray,
                     config: dict) -> Tuple[np.ndarray, np.ndarray]:
        # Override
        if self.mode == "regression" or self.mode == "multiclass":
            params = config["post_process"]["params"]
            OptR = OptimizedRounder(**params)
            OptR.fit(oof_preds, y)
            oof_preds_ = OptR.predict(oof_preds)
            test_preds_ = OptR.predict(test_preds)
            return oof_preds_, test_preds_
        return oof_preds, test_preds
