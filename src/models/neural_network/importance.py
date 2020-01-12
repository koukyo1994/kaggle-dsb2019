import numpy as np
import pandas as pd
import torch
import torch.utils.data as torchdata

from typing import List

from fastprogress import progress_bar

from .dataset import DSBDataset
from src.evaluation import OptimizedRounder, calc_metric


def permutation_importance(model: torch.nn.Module,
                           X: pd.DataFrame,
                           y: np.ndarray,
                           categorical_features: List[str],
                           mode="regression") -> np.ndarray:
    dataset = DSBDataset(df=X, categorical_features=categorical_features)
    loader = torchdata.DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=4)
    base_score = _eval_loop(model, loader, y, mode=mode)
    scores = []
    columns = X.columns
    for col in progress_bar(columns, leave=False):
        X_ = X.copy()
        X_[col] = np.random.permutation(X[col])
        dataset = DSBDataset(df=X_, categorical_features=categorical_features)
        loader = torchdata.DataLoader(
            dataset, batch_size=2048, shuffle=False, num_workers=4)
        score = _eval_loop(model, loader, y, mode=mode)
        scores.append(base_score - score)
    return np.asarray(scores)


def _eval_loop(model: torch.nn.Module,
               loader: torchdata.DataLoader,
               y: np.ndarray,
               mode="regression") -> float:
    model.eval()
    if mode in ["regression", "binary"]:
        preds = np.zeros(len(loader.dataset))
    elif mode in ["ovr", "multiclass"]:
        preds = np.zeros((len(loader.dataset), 4))

    batch_size = loader.batch_size
    for i, (non_cat, cat) in enumerate(loader):
        with torch.no_grad():
            pred = model(non_cat.float(), cat).detach()
            preds[i * batch_size:(i + 1) * batch_size] = pred.cpu().numpy()

    if mode == "multiclass":
        preds = preds @ np.arange(4) / 3
    elif mode == "ovr":
        preds = preds / np.repeat(preds.sum(axis=1), 4).reshape(-1, 4)
        preds = preds @ np.arange(4) / 3

    OptR = OptimizedRounder(n_overall=5, n_classwise=5)
    OptR.fit(preds, y)
    optimized = OptR.predict(preds)
    return calc_metric(optimized, y)
