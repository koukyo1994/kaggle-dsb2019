import lightgbm as lgb
import numpy as np
import pandas as pd

from typing import Union, Tuple

from .base2 import BaseModel2
from src.evaluation import (OptimizedRounder, lgb_classification_qwk,
                            lgb_multiclass_qwk, lgb_regression_qwk)

LGBMModel = Union[lgb.LGBMClassifier, lgb.LGBMRegressor]


class LightGBM2(BaseModel2):
    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_valid: np.ndarray, y_valid: np.ndarray,
            config: dict) -> Tuple[LGBMModel, dict]:
        model_params = config["model"]["model_params"]
        train_params = config["model"]["train_params"]

        if train_params.get("scheduler") is not None:
            train_params.pop("scheduler")

        mode = config["model"]["mode"]
        self.mode = mode
        if mode == "regression":
            self.denominator = y_train.max()
            y_train = y_train / self.denominator
            y_valid = y_valid / self.denominator

        d_train = lgb.Dataset(x_train, label=y_train)
        d_valid = lgb.Dataset(x_valid, label=y_valid)

        if mode == "regression":
            feval = lgb_regression_qwk if (mode == "regression") else None
            model = lgb.train(
                params=model_params,
                train_set=d_train,
                valid_sets=[d_valid],
                valid_names=["val"],
                feval=feval,
                **train_params)
        elif mode == "multiclass":
            model = lgb.train(
                params=model_params,
                train_set=d_train,
                valid_sets=[d_valid],
                valid_names=["val"],
                feval=lgb_multiclass_qwk,
                **train_params)
        else:
            model = lgb.train(
                params=model_params,
                train_set=d_train,
                valid_sets=[d_valid],
                valid_names=["val"],
                feval=lgb_classification_qwk,
                **train_params)
        best_score = dict(model.best_score)
        return model, best_score

    def get_best_iteration(self, model: LGBMModel) -> int:
        return model.best_iteration

    def predict(self, model: LGBMModel,
                features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.mode == "regression":
            return model.predict(features)
        elif self.mode == "multiclass":
            pred = model.predict(features) @ np.arange(4) / 3
            return pred
        else:
            return model.predict(features).reshape(4, -1).argmax(axis=0)

    def get_feature_importance(self, model: LGBMModel) -> np.ndarray:
        return model.feature_importance(importance_type="gain")

    def post_process(self, oof_preds: np.ndarray, test_preds: np.ndarray,
                     y: np.ndarray,
                     config: dict) -> Tuple[np.ndarray, np.ndarray]:
        # Override
        if (self.mode == "regression" or self.mode == "multiclass"):
            params = config["post_process"]["params"]
            OptR = OptimizedRounder(**params)
            OptR.fit(oof_preds, y)
            oof_preds_ = OptR.predict(oof_preds)
            test_preds_ = OptR.predict(test_preds)
            return oof_preds_, test_preds_
        return oof_preds, test_preds
