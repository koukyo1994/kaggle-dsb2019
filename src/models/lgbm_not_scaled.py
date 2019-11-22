import lightgbm as lgb
import numpy as np
import pandas as pd

from typing import Union, Tuple, Optional, List

from .base import BaseModel
from src.evaluation import (OptimizedRounderNotScaled,
                            lgb_regression_qwk_not_scaled,
                            lgb_classification_qwk)

LGBMModel = Union[lgb.LGBMClassifier, lgb.LGBMRegressor]


class LightGBMNotScaled(BaseModel):
    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_valid: np.ndarray, y_valid: np.ndarray,
            x_valid2: Optional[np.ndarray], y_valid2: Optional[np.ndarray],
            config: dict) -> Tuple[LGBMModel, dict]:
        model_params = config["model"]["model_params"]
        train_params = config["model"]["train_params"]

        mode = config["model"]["mode"]
        self.mode = mode

        d_train = lgb.Dataset(x_train, label=y_train)
        d_valid = lgb.Dataset(x_valid, label=y_valid)

        valid_sets: List[lgb.Dataset] = []
        valid_names: List[str] = []
        if x_valid2 is not None:
            d_valid2 = lgb.Dataset(x_valid2, label=y_valid2)
            valid_sets += [d_valid2, d_valid]
            valid_names += ["data_from_test", "data_from_train"]
        else:
            valid_sets.append(d_valid)
            valid_names.append("valid")

        if mode == "regression" or mode == "residual":
            feval = lgb_regression_qwk_not_scaled
            model = lgb.train(
                params=model_params,
                train_set=d_train,
                valid_sets=valid_sets,
                valid_names=valid_names,
                feval=feval,  # FIXME: support for residual
                **train_params)
        else:
            model = lgb.train(
                params=model_params,
                train_set=d_train,
                valid_sets=valid_sets,
                valid_names=valid_names,
                feval=lgb_classification_qwk,
                **train_params)
        best_score = dict(model.best_score)
        return model, best_score

    def get_best_iteration(self, model: LGBMModel) -> int:
        return model.best_iteration

    def predict(self, model: LGBMModel,
                features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.mode == "regression" or self.mode == "residual":
            return model.predict(features)
        else:
            return model.predict(features).reshape(4, -1).argmax(axis=0)

    def get_feature_importance(self, model: LGBMModel) -> np.ndarray:
        return model.feature_importance(importance_type="gain")

    def post_process(self, oof_preds: np.ndarray, test_preds: np.ndarray,
                     valid_preds: Optional[np.ndarray], y: np.ndarray,
                     y_valid: Optional[np.ndarray], config: dict
                     ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        # Override
        if self.mode == "regression" or self.mode == "residual":
            OptR = OptimizedRounderNotScaled()
            OptR.fit(oof_preds, y)
            coef = OptR.coefficients()
            oof_preds_ = OptR.predict(oof_preds, coef)
            test_preds_ = OptR.predict(test_preds, coef)
            if valid_preds is not None:
                valid_preds = OptR.predict(valid_preds, coef)
            return oof_preds_, test_preds_, valid_preds
        return oof_preds, test_preds, valid_preds
