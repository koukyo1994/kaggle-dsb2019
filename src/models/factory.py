from .cat import CatBoostModel
from .lgbm import LightGBM
from .classwise_cat import ClassWiseCatBoost
from .classwise_lgbm import ClassWiseLightGBM


def lgbm() -> LightGBM:
    return LightGBM()


def catboost() -> CatBoostModel:
    return CatBoostModel()


def classwise_cat() -> ClassWiseCatBoost:
    return ClassWiseCatBoost()


def classwise_lgbm() -> ClassWiseLightGBM:
    return ClassWiseLightGBM()


def get_model(config: dict):
    model_name = config["model"]["name"]
    func = globals().get(model_name)
    if func is None:
        raise NotImplementedError
    return func()
