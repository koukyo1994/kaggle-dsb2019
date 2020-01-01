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


def truncated_cv_with_adjustment_of_distribution(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        groups: np.ndarray,
        test_groups: np.ndarray,
        n_trials: int = 100) -> Dict[str, Union[List[float], float]]:

    eval_result: Dict[str, Union[List[float], float]] = {}
    trials: List[float] = []

    first_assess_ratio = len(test_groups[test_groups == 1]) / len(test_groups)
    second_assess_ratio = len(test_groups[test_groups == 2]) / len(test_groups)
    over_third_ratio = len(test_groups[test_groups >= 3]) / len(test_groups)

    index = np.arange(len(groups))

    first_assess_idx = []
    second_assess_idx = []
    over_third_assess_idx = []

    for inst_id in np.unique(groups):
        idx_inst = index[groups == inst_id]
        first_assess_idx.append(idx_inst[0])
        if len(idx_inst) > 1:
            second_assess_idx.append(idx_inst[1])

        if len(idx_inst) > 2:
            over_third_assess_idx.append(idx_inst[2])

    total_assess = len(groups)
    for i in np.arange(1.0, 0.0, -0.1):
        n_assess = int(total_assess * i)
        n_first_assess = int(first_assess_ratio * n_assess)
        n_second_assess = int(second_assess_ratio * n_assess)
        n_over_third_assess = int(over_third_ratio * n_assess)

        if (n_first_assess < len(first_assess_idx)
                and n_second_assess < len(second_assess_idx)
                and n_over_third_assess < len(over_third_assess_idx)):
            break

    print("n_first_assess: {} ({:.3f} %)".format(
        n_first_assess, n_first_assess / n_assess * 100))
    print("n_second_assess: {} ({:.3f} %)".format(
        n_second_assess, n_second_assess / n_assess * 100))
    print("n_over_assess: {} ({:.3f} %)".format(
        n_over_third_assess, n_over_third_assess / n_assess * 100))

    first_assess_dice_result = []
    second_assess_dice_result = []
    over_third_dice_result = []
    for _ in range(n_trials):
        first_assess_dice_result.append(
            np.random.choice(
                first_assess_idx, size=n_first_assess, replace=False))
        second_assess_dice_result.append(
            np.random.choice(
                second_assess_idx, size=n_second_assess, replace=False))
        over_third_dice_result.append(
            np.random.choice(
                over_third_assess_idx, size=n_over_third_assess,
                replace=False))

    first_assess = np.asarray(first_assess_dice_result).T
    second_assess = np.asarray(second_assess_dice_result).T
    over_third_assess = np.asarray(over_third_dice_result).T

    assess = np.vstack([first_assess, second_assess, over_third_assess])

    for i in range(n_trials):
        y_pred_choice = y_pred[assess[:, i]]
        y_true_choice = y_true[assess[:, i]]
        trials.append(calc_metric(y_true_choice, y_pred_choice))

    mean_score = np.mean(trials)
    std = np.std(trials)
    eval_result["mean"] = mean_score
    eval_result["all_trials"] = trials
    eval_result["0.95lower_bound"] = mean_score - 2 * std
    eval_result["0.95upper_bound"] = mean_score + 2 * std
    eval_result["std"] = std
    return eval_result
