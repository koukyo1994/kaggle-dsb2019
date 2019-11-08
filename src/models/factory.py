from .cat import CatBoost
from .lgbm import LightGBM


def lgbm() -> LightGBM:
    return LightGBM()


def catboost() -> CatBoost:
    return CatBoost()


def get_model(config: dict):
    model_name = config["model"]["name"]
    func = globals().get(model_name)
    if func is None:
        raise NotImplementedError
    return func()
