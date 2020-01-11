import numpy as np
import pandas as pd
import scipy as sp

from functools import partial

from src.evaluation import calc_metric


class GroupWiseOptimizer(object):
    def __init__(self, n_overall: int = 5, n_classwise: int = 5):
        self.n_overall = n_overall
        self.n_classwise = n_classwise

    def fit(self, X: np.ndarray, y: np.ndarray, group: np.ndarray):
        self.rounders = {
            gp: OptimizedRounder(
                n_overall=self.n_overall, n_classwise=self.n_classwise)
            for gp in np.unique(group)
        }
        for gp in self.rounders.keys():
            X_gp = X[group == gp]
            y_gp = y[group == gp]
            self.rounders[gp].fit(X_gp, y_gp)

    def predict(self, X: np.ndarray, group: np.ndarray) -> np.ndarray:
        result = np.zeros_like(X)
        for gp in self.rounders.keys():
            X_gp = X[group == gp]
            result[group == gp] = self.rounders[gp].predict(X_gp)
        return result


class OptimizedRounder(object):
    def __init__(self,
                 n_overall: int = 5,
                 n_classwise: int = 5,
                 reverse: bool = False):
        self.n_overall = n_overall
        self.n_classwise = n_classwise
        self.coef = [0.25, 0.5, 0.75]
        self.reverse = reverse

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        X_p = np.digitize(X, self.coef)
        ll = -calc_metric(y, X_p)
        return ll

    def fit(self, X: np.ndarray, y: np.ndarray):
        golden1 = 0.618
        golden2 = 1 - golden1
        ab_start = [(0.01, 0.5), (0.5, 0.7), (0.7, 0.9)]
        for _ in range(self.n_overall):
            if self.reverse:
                search = reversed(range(3))
            else:
                search = iter(range(3))
            for idx in search:
                # golden section search
                a, b = ab_start[idx]
                # calc losses
                self.coef[idx] = a
                la = self._loss(X, y)
                self.coef[idx] = b
                lb = self._loss(X, y)
                for it in range(self.n_classwise):
                    # choose value
                    if la > lb:
                        a = b - (b - a) * golden1
                        self.coef[idx] = a
                        la = self._loss(X, y)
                    else:
                        b = b - (b - a) * golden2
                        self.coef[idx] = b
                        lb = self._loss(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_p = np.digitize(X, self.coef)
        return X_p


class OptimizedRounderNotScaled(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = pd.cut(
            X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3])

        return -calc_metric(y, X_p)

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = sp.optimize.minimize(
            loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        return pd.cut(
            X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3])

    def coefficients(self):
        return self.coef_['x']
