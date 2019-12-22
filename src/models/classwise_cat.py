import numpy as np
import pandas as pd

from typing import Tuple, Union, List

from catboost import CatBoostClassifier, CatBoostRegressor

from .classwise import ClassWiseBase
from src.evaluation import (CatBoostOptimizedQWKMetric, OptimizedRounder,
                            CatBoostMulticlassOptimizedQWK)

CatModel = Union[CatBoostClassifier, CatBoostRegressor]


class ClassWiseCatBoost(ClassWiseBase):
    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            valid_sets: List[Tuple[np.ndarray, np.ndarray]],
            config: dict) -> Tuple[CatModel, dict]:
        model_params = config["model"]["model_params"]
        mode = config["model"]["mode"]
        self.mode = mode

        if mode == "regression":
            model = CatBoostRegressor(
                eval_metric=CatBoostOptimizedQWKMetric(
                    reverse=config["post_process"]["params"]["reverse"]),
                **model_params)
            self.denominator = y_train.max()
            y_train = y_train / y_train.max()
            eval_sets = []
            for x, y in valid_sets:
                eval_sets.append((x, y / self.denominator))
        elif mode == "multiclass":
            model = CatBoostClassifier(
                eval_metric=CatBoostMulticlassOptimizedQWK(
                    reverse=config["post_process"]["params"]["reverse"]),
                **model_params)
        else:
            model = CatBoostClassifier(**model_params)
            eval_sets = valid_sets

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
        if self.mode != "multiclass":
            return model.predict(features)
        else:
            preds = model.predict_proba(features)
            return preds @ np.arange(4) / 3

    def get_feature_importance(self, model: CatModel) -> np.ndarray:
        return model.feature_importances_

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
