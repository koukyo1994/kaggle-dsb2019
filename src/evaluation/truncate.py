import numpy as np
import pandas as pd

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
    gp_idx_df = pd.DataFrame({"groups": groups, "index": index})
    dice_results = []
    for _, df in gp_idx_df.groupby("groups"):
        dice_result = np.random.choice(df["index"], size=n_trials)
        dice_results.append(dice_result)

    idx_choice = np.vstack(dice_results)
    for i in range(n_trials):
        y_pred_choice = y_pred[idx_choice[:, i]]
        y_true_choice = y_true[idx_choice[:, i]]
        trials.append(calc_metric(y_true_choice, y_pred_choice))

    mean_score = np.mean(trials)
    std = np.std(trials)
    eval_result["mean"] = mean_score
    eval_result["all_trials"] = trials
    eval_result["0.95lower_bound"] = mean_score - 2 * std
    eval_result["0.95upper_bound"] = mean_score + 2 * std
    eval_result["std"] = std
    return eval_result
