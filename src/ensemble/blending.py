import numpy as np

from typing import List, Callable

from src.utils import seed_everything


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(((a - b)**2).mean())


def search_blending_weight(
        predictions: List[np.ndarray],
        target: np.ndarray,
        n_iter: int,
        func: Callable[[np.ndarray, np.ndarray], float] = rmse,
        is_higher_better: bool = False) -> np.ndarray:
    best_weights = np.zeros(len(predictions))
    best_score = -np.inf if is_higher_better else np.inf

    for i in range(n_iter):
        seed_everything(i)
        dice = np.random.rand(len(predictions))
        weights = dice / dice.sum()
        blended = np.zeros(len(predictions[0]))
        for weight, pred in zip(weights, predictions):
            blended += weight * pred
        score = func(blended, target)
        if is_higher_better:
            if score > best_score:
                best_score = score
                best_weights = weights
        else:
            if score < best_score:
                best_score = score
                best_weights = weights
    return best_score, best_weights
