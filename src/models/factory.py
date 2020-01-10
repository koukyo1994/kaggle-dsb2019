from .cat import CatBoostModel
from .lgbm import LightGBM
from .classwise_cat import ClassWiseCatBoost
from .classwise_lgbm import ClassWiseLightGBM
from .cat_not_scaled import CatBoostNotScaledModel
from .lgbm_not_scaled import LightGBMNotScaled
from .lgbm2 import LightGBM2
from .cat2 import CatBoostModel2
from .weighted_lgbm import WeightedLightGBM
from .xgb2 import XGBoost2
from .nn import NNTrainer


def lgbm() -> LightGBM:
    return LightGBM()


def lgbm2() -> LightGBM2:
    return LightGBM2()


def weighted_lgbm() -> WeightedLightGBM:
    return WeightedLightGBM()


def catboost() -> CatBoostModel:
    return CatBoostModel()


def catboost2() -> CatBoostModel2:
    return CatBoostModel2()


def nn() -> NNTrainer:
    return NNTrainer()


def classwise_cat() -> ClassWiseCatBoost:
    return ClassWiseCatBoost()


def classwise_lgbm() -> ClassWiseLightGBM:
    return ClassWiseLightGBM()


def catboost_not_scaled() -> CatBoostNotScaledModel:
    return CatBoostNotScaledModel()


def lgbm_not_scaled() -> LightGBMNotScaled:
    return LightGBMNotScaled()


def xgb2() -> XGBoost2:
    return XGBoost2()


def get_model(config: dict):
    model_name = config["model"]["name"]
    func = globals().get(model_name)
    if func is None:
        raise NotImplementedError
    return func()
