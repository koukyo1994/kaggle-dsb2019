import logging

import catboost as cat
import lightgbm as lgb
import numpy as np
import pandas as pd

from abc import abstractmethod
from typing import Dict, List, Union, Tuple

from src.evaluation import calc_metric

# type alias
AoD = Union[np.ndarray, pd.DataFrame]
AoS = Union[np.ndarray, pd.Series]
CatModel = Union[cat.CatBoostClassifier, cat.CatBoostRegressor]
LGBModel = Union[lgb.LGBMClassifier, lgb.LGBMRegressor]
Model = Union[CatModel, LGBModel]


class ClassWiseBase(object):
    @abstractmethod
    def fit(self, x_train: AoD, y_train: AoS,
            valid_sets: List[Tuple[AoD, AoS]],
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

    def post_process(self, preds_set: List[Tuple[np.ndarray, np.ndarray]],
                     test_preds: np.ndarray,
                     config: dict) -> Tuple[List[np.ndarray], np.ndarray]:
        return [p[0] for p in preds_set], test_preds

    def cv(self,
           y_train: AoS,
           train_features: AoD,
           test_features: AoD,
           y_valid: AoS,
           valid_features: AoD,
           feature_name: List[str],
           folds_ids: List[Tuple[np.ndarray, np.ndarray]],
           config: dict,
           log: bool = True) -> Tuple[Dict[int, List[Model]], np.ndarray, np.
                                      ndarray, np.ndarray, dict, dict]:
        # initialize
        test_preds = np.zeros(len(test_features))
        oof_preds = np.zeros(len(train_features))
        valid_preds = np.zeros(len(valid_features))

        all_classes = train_features["session_title"].unique()

        classwise_mean_importances = {}
        classwise_best_iteration = {c: 0.0 for c in all_classes}
        classwise_cv_score_list = {c: [] for c in all_classes}
        classwise_models = {c: [] for c in all_classes}

        X = train_features.values if isinstance(train_features, pd.DataFrame) \
            else train_features
        y = y_train.values if isinstance(y_train, pd.Series) else y_train

        X_valid = valid_features.values if isinstance(valid_features, pd.DataFrame) \
            else valid_features
        y_valid = y_valid.values if isinstance(y_valid, pd.Series) else y_valid

        for c in all_classes:
            importances = pd.DataFrame(index=feature_name)
            train_c_idx = train_features.query(
                f"session_title == {c}").index.values
            valid_c_idx = valid_features.query(
                f"session_title == {c}").index.values
            test_c_idx = test_features.query(
                f"session_title == {c}").index.values

            X_c_valid = X_valid[valid_c_idx]
            y_c_valid = y_valid[valid_c_idx]

            print(f"Assessment Class: {c}")

            for i_fold, (trn_idx, val_idx) in enumerate(folds_ids):
                print("=" * 15)
                print(f"Fold: {i_fold + 1}")
                print("=" * 15)

                trn_c_idx = np.intersect1d(trn_idx, train_c_idx)
                val_c_idx = np.intersect1d(val_idx, train_c_idx)

                # get train data and valid data
                X_trn = X[trn_c_idx]
                y_trn = y[trn_c_idx]
                X_val = X[val_c_idx]
                y_val = y[val_c_idx]

                # train model
                model, best_score = self.fit(
                    X_trn,
                    y_trn,
                    valid_sets=[(X_val, y_val), (X_c_valid, y_c_valid)],
                    config=config)
                classwise_cv_score_list[c].append(best_score)
                classwise_models[c].append(model)
                classwise_best_iteration[c] += self.get_best_iteration(model)

                # predict oof and test, valid
                oof_preds[val_c_idx] = self.predict(model, X_val).reshape(-1)
                test_preds[test_c_idx] += self.predict(
                    model, test_features.loc[test_c_idx, :]).reshape(-1) / len(
                        folds_ids)
                valid_preds[valid_c_idx] += self.predict(
                    model, X_c_valid).reshape(-1) / len(folds_ids)

                # get feature importances
                importances_tmp = pd.DataFrame(
                    self.get_feature_importance(model),
                    columns=[f"class_{c}_gain_{i_fold+1}"],
                    index=feature_name)
                importances = importances.join(importances_tmp, how="inner")

            # summary of feature importance
            classwise_mean_importances[c] = importances.mean(axis=1)

        # save raw prediction
        self.raw_oof_preds = oof_preds
        self.raw_test_preds = test_preds
        self.raw_valid_preds = valid_preds

        # post_process
        [oof_preds, valid_preds], test_preds = self.post_process(
            [(oof_preds, y_train), (valid_preds, y_valid)], test_preds, config)

        # print oof score
        oof_score = calc_metric(y_train, oof_preds)
        print(f"oof score: {oof_score:.5f}")
        valid_score = calc_metric(y_valid, valid_preds)
        print(f"valid score: {valid_score:.5f}")

        if log:
            logging.info(f"oof score: {oof_score:.5f}")
            logging.info(f"valid score: {valid_score:.5f}")

        eval_results = {
            "eval_result": {
                "oof_score": oof_score,
                "valid_score": valid_score,
                "cv_results": {},
                "n_data": len(train_features),
                "n_features": len(train_features.columns),
                "best_iterations": {
                    f"Assessment {c}": v
                    for c, v in classwise_best_iteration.items()
                },
                "feature_importances": {}
            }
        }

        for c, v in classwise_cv_score_list.items():
            eval_results["eval_result"]["cv_results"][f"Assessment {c}"] = \
                {f"cv{i + 1}": cv_score for i, cv_score in enumerate(v)}

        for c, fi in classwise_mean_importances.items():
            eval_results["eval_result"]["feature_importances"][
                f"Assessment {c}"] = fi.sort_values(ascending=False).to_dict()

        return (classwise_models, oof_preds, test_preds, valid_preds,
                classwise_mean_importances, eval_results)
