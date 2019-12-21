import lightgbm as lgb
import numpy as np

from typing import Tuple

from .metrics import calc_metric
from .optimization import OptimizedRounder, OptimizedRounderNotScaled


def lgb_classification_qwk(y_pred: np.ndarray,
                           data: lgb.Dataset) -> Tuple[str, float, bool]:
    y_true = data.get_label()
    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    return "qwk", calc_metric(y_true, y_pred), True


def lgb_multiclass_qwk(y_pred: np.ndarray,
                       data: lgb.Dataset) -> Tuple[str, float, bool]:
    y_true = data.get_label()
    y_pred = y_pred.reshape(len(np.unique(y_true)), -1)
    y_pred_regr = np.arange(4) @ y_pred / 3

    OptR = OptimizedRounder(n_classwise=3, n_overall=3)
    OptR.fit(y_pred_regr, y_true)

    y_pred = OptR.predict(y_pred_regr).astype(int)
    return "qwk", calc_metric(y_true, y_pred), True


def lgb_regression_qwk(y_pred: np.ndarray,
                       data: lgb.Dataset) -> Tuple[str, float, bool]:
    y_true = (data.get_label() * 3).astype(int)
    y_pred = y_pred.reshape(-1)

    OptR = OptimizedRounder(n_classwise=5, n_overall=5)
    OptR.fit(y_pred, y_true)

    y_pred = OptR.predict(y_pred).astype(int)
    qwk = calc_metric(y_true, y_pred)

    return "qwk", qwk, True


def lgb_regression_qwk_not_scaled(
        y_pred: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    y_true = data.get_label().astype(int)
    y_pred = y_pred.reshape(-1)

    OptR = OptimizedRounderNotScaled()
    OptR.fit(y_pred, y_true)

    coef = OptR.coefficients()

    y_pred = OptR.predict(y_pred, coef).astype(int)
    qwk = calc_metric(y_true, y_pred)

    return "qwk", qwk, True


def lgb_residual_qwk_closure(mean_target: np.ndarray):
    def lgb_residual_qwk(y_pred: np.ndarray,
                         data: lgb.Dataset) -> Tuple[str, float, bool]:
        y_true = (data.get_label() * 3).astype(int)
        y_pred = y_pred.reshape(-1)

        y_true = (y_true + mean_target).astype(int)
        y_pred = y_pred + mean_target

        OptR = OptimizedRounder(n_classwise=5, n_overall=5)
        OptR.fit(y_pred, y_true)

        y_pred = OptR.predict(y_pred).astype(int)
        qwk = calc_metric(y_true, y_pred)

        return "qwk", qwk, True

    return lgb_residual_qwk
