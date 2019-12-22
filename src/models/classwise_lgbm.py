import lightgbm as lgb
import numpy as np
import pandas as pd

from typing import Union, Tuple, List

from .classwise import ClassWiseBase
from src.evaluation import (OptimizedRounder, lgb_classification_qwk,
                            lgb_regression_qwk, lgb_multiclass_qwk)

LGBMModel = Union[lgb.LGBMClassifier, lgb.LGBMRegressor]


class ClassWiseLightGBM(ClassWiseBase):
    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            valid_sets: List[Tuple[np.ndarray, np.ndarray]],
            config: dict) -> Tuple[LGBMModel, dict]:
        model_params = config["model"]["model_params"]
        train_params = config["model"]["train_params"]

        mode = config["model"]["mode"]
        self.mode = mode
        if mode == "regression":
            self.denominator = y_train.max()
            y_train = y_train / self.denominator
            eval_sets = []
            for x, y in valid_sets:
                eval_sets.append(lgb.Dataset(x, label=y / self.denominator))
        elif mode == "multiclass":
            eval_sets = []
            for x, y in valid_sets:
                eval_sets.append(lgb.Dataset(x, label=y))

        d_train = lgb.Dataset(x_train, label=y_train)

        if mode == "regression":
            feval = lgb_regression_qwk
        elif mode == "multiclass":
            feval = lgb_multiclass_qwk
        else:
            feval = lgb_classification_qwk

        model = lgb.train(
            params=model_params,
            train_set=d_train,
            valid_sets=eval_sets,
            feval=feval,
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

    def post_process(
            self, preds_set: List[Tuple[np.ndarray, np.ndarray]],
            test_preds: np.ndarray, config: dict
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        # Override
        if self.mode == "regression" or self.mode == "multiclass":
            params = config["post_process"]["params"]
            OptR = OptimizedRounder(**params)
            OptR.fit(preds_set[0][0], preds_set[0][1])

            return_set = [OptR.predict(l[0]) for l in preds_set]
            test_preds = OptR.predict(test_preds)
            return return_set, test_preds
        return preds_set, test_preds
