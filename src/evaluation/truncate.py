import numpy as np

from typing import Dict, List, Union

from .metrics import calc_metric


def eval_with_truncated_data(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        groups: np.ndarray,
        n_trials: int = 10) -> Dict[str, Union[List[float], float]]:
    eval_result: Dict[str, Union[List[float], float]] = {}
    trials: List[float] = []

    index = np.arange(len(y_pred))
    for _ in range(n_trials):
        idx_choice = []
        for group in np.unique(groups):
            grp_idx = index[groups == group]
            idx_choice.append(np.random.choice(grp_idx))
        y_pred_choice = y_pred[idx_choice]
        y_true_choice = y_true[idx_choice]

        score = calc_metric(y_true_choice, y_pred_choice)
        trials.append(score)
    mean_score = np.mean(trials)
    eval_result["mean"] = mean_score
    eval_result["all_trials"] = trials
    return eval_result
