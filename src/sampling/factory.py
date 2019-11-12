import numpy as np

from typing import Tuple

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def smote(x_trn: np.ndarray, y_trn: np.ndarray,
          config: dict) -> Tuple[np.ndarray, np.ndarray]:
    params = config["model"]["sampling"]["params"]
    sm = SMOTE(
        k_neighbors=params["k_neighbors"], random_state=params["random_state"])
    sampled_x, sampled_y = sm.fit_resample(x_trn, y_trn)
    return sampled_x, sampled_y


def random_under_sample(x_trn: np.ndarray, y_trn: np.ndarray,
                        config: dict) -> Tuple[np.ndarray, np.ndarray]:
    params = config["model"]["sampling"]["params"]
    acc_0 = (y_trn == 0).sum().astype(int)
    acc_1 = (y_trn == 1).sum().astype(int)
    acc_2 = (y_trn == 2).sum().astype(int)
    acc_3 = (y_trn == 3).sum().astype(int)
    rus = RandomUnderSampler({
        0: int(params["acc_0_coef"] * acc_0),
        1: int(params["acc_1_coef"] * acc_1),
        2: int(params["acc_2_coef"] * acc_2),
        3: int(params["acc_3_coef"] * acc_3)
    },
                             random_state=params["random_state"])
    sampled_x, sampled_y = rus.fit_resample(x_trn, y_trn)
    return sampled_x, sampled_y


def random_under_sample_and_smote(
        x_trn: np.ndarray, y_trn: np.ndarray,
        config: dict) -> Tuple[np.ndarray, np.ndarray]:
    sampled_x, sampled_y = random_under_sample(x_trn, y_trn, config)
    sampled_x, sampled_y = smote(sampled_x, sampled_y, config)
    return sampled_x, sampled_y


def get_sampling(x_trn: np.ndarray, y_trn: np.ndarray,
                 config: dict) -> Tuple[np.ndarray, np.ndarray]:
    if config["model"]["sampling"]["name"] == "none":
        return x_trn, y_trn

    policy = config["model"]["sampling"]["name"]
    func = globals().get(policy)
    if func is None:
        raise NotImplementedError
    return func(x_trn, y_trn, config)
