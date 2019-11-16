import logging

import catboost as cat
import lightgbm as lgb
import numpy as np
import pandas as pd

from abc import abstractmethod
from typing import Dict, Union, Tuple, List, Optional

from src.evaluation import calc_metric
from src.sampling import get_sampling

# type alias
AoD = Union[np.ndarray, pd.DataFrame]
AoS = Union[np.ndarray, pd.Series]
CatModel = Union[cat.CatBoostClassifier, cat.CatBoostRegressor]
LGBModel = Union[lgb.LGBMClassifier, lgb.LGBMRegressor]
Model = Union[CatModel, LGBModel]


class BaseModel(object):
    @abstractmethod
    def fit(self, x_train: AoD, y_train: AoS, x_valid: AoD, y_valid: AoS,
            x_valid2: Optional[AoD], y_valid2: Optional[AoS],
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
                     valid_preds: Optional[np.ndarray], y: np.ndarray,
                     y_valid: Optional[np.ndarray], config: dict
                     ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        return oof_preds, test_preds, valid_preds

    def cv(self,
           y_train: AoS,
           train_features: AoD,
           test_features: AoD,
           y_valid: Optional[AoS],
           valid_features: Optional[AoD],
           feature_name: List[str],
           folds_ids: List[Tuple[np.ndarray, np.ndarray]],
           config: dict,
           log: bool = True
           ) -> Tuple[List[Model], np.ndarray, np.
                      ndarray, Optional[np.ndarray], pd.DataFrame, dict]:
        # initialize
        test_preds = np.zeros(len(test_features))
        oof_preds = np.zeros(len(train_features))
        if valid_features is not None:
            valid_preds = np.zeros(len(valid_features))
        else:
            valid_preds = None
        importances = pd.DataFrame(index=feature_name)
        best_iteration = 0.0
        cv_score_list: List[dict] = []
        models: List[Model] = []

        if config["model"]["mode"] == "residual":
            self.mean_targets: Dict[str, List[np.ndarray]] = {
                "train": [],
                "valid": [],
                "valid2": [valid_features["mean_target"].values],
                "test": [test_features["mean_target"].values]
            }
            valid_features.drop("mean_target", axis=1, inplace=True)
            test_features.drop("mean_target", axis=1, inplace=True)
            for t_idx, v_idx in folds_ids:
                self.mean_targets["train"].append(
                    train_features.loc[t_idx, "mean_target"].values)
                self.mean_targets["valid"].append(
                    train_features.loc[v_idx, "mean_target"].values)
            train_features.drop("mean_target", axis=1, inplace=True)
            feature_name.remove("mean_target")

        X = train_features.values if isinstance(train_features, pd.DataFrame) \
            else train_features
        y = y_train.values if isinstance(y_train, pd.Series) \
            else y_train

        X_valid = valid_features.values if isinstance(
            valid_features, pd.DataFrame) else valid_features
        y_valid = y_valid.values if isinstance(y_valid, pd.Series) \
            else y_valid

        for i_fold, (trn_idx, val_idx) in enumerate(folds_ids):
            self.fold = i_fold
            # get train data and valid data
            x_trn = X[trn_idx]
            y_trn = y[trn_idx]
            x_val = X[val_idx]
            y_val = y[val_idx]

            x_trn, y_trn = get_sampling(x_trn, y_trn, config)

            # train model
            model, best_score = self.fit(
                x_trn, y_trn, x_val, y_val, X_valid, y_valid, config=config)
            cv_score_list.append(best_score)
            models.append(model)
            best_iteration += self.get_best_iteration(model) / len(folds_ids)

            # predict oof and test
            oof_preds[val_idx] = self.predict(model, x_val).reshape(-1)
            test_preds += self.predict(
                model, test_features).reshape(-1) / len(folds_ids)

            if valid_features is not None:
                valid_preds += self.predict(
                    model, valid_features).reshape(-1) / len(folds_ids)

            if config["model"]["mode"] == "residual":
                oof_preds[val_idx] += self.mean_targets["valid"][self.fold]
                test_preds += self.mean_targets["test"][0]
                valid_preds += self.mean_targets["valid2"][0]

            # get feature importances
            importances_tmp = pd.DataFrame(
                self.get_feature_importance(model),
                columns=[f"gain_{i_fold+1}"],
                index=feature_name)
            importances = importances.join(importances_tmp, how="inner")

        # summary of feature importance
        feature_importance = importances.mean(axis=1)

        # save raw prediction
        self.raw_oof_preds = oof_preds
        self.raw_test_preds = test_preds
        self.raw_valid_preds = valid_preds

        # post_process (if you have any)
        oof_preds, test_preds, valid_preds = self.post_process(
            oof_preds, test_preds, valid_preds, y_train, y_valid, config)

        # print oof score
        oof_score = calc_metric(y_train, oof_preds)
        print(f"oof score: {oof_score:.5f}")
        if valid_features is not None:
            valid_score = calc_metric(y_valid, valid_preds)
            print(f"valid score: {valid_score:.5f}")

        if log:
            logging.info(f"oof score: {oof_score:.5f}")
            if valid_features is not None:
                logging.info(f"valid score: {valid_score:.5f}")

        evals_results = {
            "evals_result": {
                "oof_score":
                oof_score,
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

        if valid_features is not None:
            evals_results["valid_score"] = valid_score
        return (models, oof_preds, test_preds, valid_preds, feature_importance,
                evals_results)
