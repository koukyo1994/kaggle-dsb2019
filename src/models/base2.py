import logging

import catboost as cat
import lightgbm as lgb
import numpy as np
import pandas as pd

from abc import abstractmethod
from typing import Union, Tuple, List

from src.evaluation import (calc_metric, eval_with_truncated_data,
                            OptimizedRounder)

# type alias
AoD = Union[np.ndarray, pd.DataFrame]
AoS = Union[np.ndarray, pd.Series]
CatModel = Union[cat.CatBoostClassifier, cat.CatBoostRegressor]
LGBModel = Union[lgb.LGBMClassifier, lgb.LGBMRegressor]
Model = Union[CatModel, LGBModel]


class BaseModel2(object):
    @abstractmethod
    def fit(self, x_train: AoD, y_train: AoS, x_valid: AoD, y_valid: AoS,
            config: dict) -> Tuple[Model, dict]:
        raise NotImplementedError

    @abstractmethod
    def get_best_iteration(self, model: Model) -> int:
        raise NotImplementedError

    @abstractmethod
    def predict(self, model: Model, features: AoD) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_feature_importance(self, model: Model) -> np.ndarray:
        raise NotImplementedError

    def post_process(self, oof_preds: np.ndarray, test_preds: np.ndarray,
                     y: np.ndarray,
                     config: dict) -> Tuple[np.ndarray, np.ndarray]:
        return oof_preds, test_preds

    def cv(self,
           y_train: AoS,
           train_features: AoD,
           test_features: AoD,
           groups: np.ndarray,
           feature_name: List[str],
           folds_ids: List[Tuple[np.ndarray, np.ndarray]],
           threshold: float,
           config: dict,
           log: bool = True) -> Tuple[List[Model], np.ndarray, np.ndarray, np.
                                      ndarray, pd.DataFrame, dict]:
        # initialize
        test_preds = np.zeros(len(test_features))
        normal_oof_preds = np.zeros(len(train_features))
        oof_preds_list: List[np.ndarray] = []
        y_val_list: List[np.ndarray] = []

        importances = pd.DataFrame(index=feature_name)
        best_iteration = 0.0
        cv_score_list: List[dict] = []
        models: List[Model] = []

        X = train_features.values if isinstance(train_features, pd.DataFrame) \
            else train_features
        y = y_train.values if isinstance(y_train, pd.Series) \
            else y_train

        for i_fold, (trn_idx, val_idx) in enumerate(folds_ids):
            self.fold = i_fold
            # get train data and valid data
            x_trn = X[trn_idx]
            y_trn = y[trn_idx]

            val_idx = np.sort(val_idx)
            gp_val = groups[val_idx]

            new_idx: List[int] = []
            for gp in np.unique(gp_val):
                gp_idx = val_idx[gp_val == gp]
                new_idx.extend(gp_idx[:int(threshold)])

            x_val = X[new_idx]
            y_val = y[new_idx]
            x_val_normal = X[val_idx]

            # train model
            model, best_score = self.fit(
                x_trn, y_trn, x_val, y_val, config=config)
            cv_score_list.append(best_score)
            models.append(model)
            best_iteration += self.get_best_iteration(model) / len(folds_ids)

            # predict oof and test
            oof_preds_list.append(self.predict(model, x_val).reshape(-1))
            y_val_list.append(y_val)
            normal_oof_preds[val_idx] = self.predict(model,
                                                     x_val_normal).reshape(-1)
            test_preds += self.predict(
                model, test_features).reshape(-1) / len(folds_ids)

            # get feature importances
            importances_tmp = pd.DataFrame(
                self.get_feature_importance(model),
                columns=[f"gain_{i_fold+1}"],
                index=feature_name)
            importances = importances.join(importances_tmp, how="inner")

        # summary of feature importance
        feature_importance = importances.mean(axis=1)

        oof_preds = np.concatenate(oof_preds_list)
        y_oof = np.concatenate(y_val_list)

        # save raw prediction
        self.raw_oof_preds = oof_preds
        self.raw_test_preds = test_preds

        # post_process (if you have any)
        oof_preds, test_preds = self.post_process(oof_preds, test_preds, y_oof,
                                                  config)

        # print oof score
        oof_score = calc_metric(y_oof, oof_preds)
        print(f"oof score: {oof_score:.5f}")

        # normal oof score
        OptR = OptimizedRounder()
        OptR.fit(normal_oof_preds, y_train)
        normal_oof_preds = OptR.predict(normal_oof_preds)
        normal_oof_score = calc_metric(y_train, normal_oof_preds)
        print(f"normal oof score: {normal_oof_score:.5f}")

        # truncated score
        truncated_result = eval_with_truncated_data(
            normal_oof_preds, y_train, groups, n_trials=100)
        print(f"truncated mean: {truncated_result['mean']:.5f}")
        print(f"truncated std: {truncated_result['std']:.5f}")

        if log:
            logging.info(f"oof score: {oof_score:.5f}")
            logging.info(f"normal oof score: {normal_oof_score:.5f}")

        evals_results = {
            "evals_result": {
                "oof_score":
                oof_score,
                "normal_oof_score":
                normal_oof_score,
                "truncated_eval_mean":
                truncated_result["mean"],
                "truncated_eval_0.95upper":
                truncated_result["0.95upper_bound"],
                "truncated_eval_0.95lower":
                truncated_result["0.95lower_bound"],
                "truncated_eval_std":
                truncated_result["std"],
                "cv_score": {
                    f"cv{i + 1}": cv_score
                    for i, cv_score in enumerate(cv_score_list)
                },
                "n_data":
                len(train_features),
                "best_iteration":
                best_iteration,
                "n_features":
                len(train_features.columns),
                "feature_importance":
                feature_importance.sort_values(ascending=False).to_dict()
            }
        }

        return (models, oof_preds, y_oof, test_preds, feature_importance,
                evals_results)
