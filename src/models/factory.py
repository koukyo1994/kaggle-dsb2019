from .cat import CatBoostModel
from .lgbm import LightGBM
from .classwise_cat import ClassWiseCatBoost
from .classwise_lgbm import ClassWiseLightGBM
from .cat_not_scaled import CatBoostNotScaledModel
from .lgbm_not_scaled import LightGBMNotScaled


def lgbm() -> LightGBM:
    return LightGBM()


def catboost() -> CatBoostModel:
    return CatBoostModel()


def classwise_cat() -> ClassWiseCatBoost:
    return ClassWiseCatBoost()


def classwise_lgbm() -> ClassWiseLightGBM:
    return ClassWiseLightGBM()


def catboost_not_scaled() -> CatBoostNotScaledModel:
    return CatBoostNotScaledModel()


def lgbm_not_scaled() -> LightGBMNotScaled:
    return LightGBMNotScaled()


def get_model(config: dict):
    model_name = config["model"]["name"]
    func = globals().get(model_name)
    if func is None:
        raise NotImplementedError
    return func()
