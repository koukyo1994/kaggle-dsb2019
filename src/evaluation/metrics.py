import numpy as np

from typing import Union


def calc_metric(y_true: Union[np.ndarray, list],
                y_pred: Union[np.ndarray, list]) -> float:
    raise NotImplementedError
