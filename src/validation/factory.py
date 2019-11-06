import numpy as np
import pandas as pd

from typing import List, Tuple

from easydict import EasyDict as edict

from sklearn.model_selection import GroupKFold


def group_kfold(df: pd.DataFrame, groups: pd.Series,
                config: edict) -> List[Tuple[np.ndarray, np.ndarray]]:
    params = config.val.params
    kf = GroupKFold(n_splits=params.n_splits)
    split = list(kf.split(df, groups=groups))
    return split


def get_validation(df: pd.DataFrame,
                   config: edict) -> List[Tuple[np.ndarray, np.ndarray]]:
    name: str = config.val.name

    func = globals().get(name)
    if func is None:
        raise NotImplementedError

    if "group" in name:
        cols = df.columns.tolist()
        cols.remove("group")
        groups = df["group"]
        return func(df[cols], groups, config)
    else:
        return func(df, config)
