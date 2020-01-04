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

    n_first_assess = len(test_groups[test_groups == 1])
    n_second_assess = len(test_groups[test_groups == 2])
    n_third_assess = len(test_groups[test_groups == 3])
    n_fourth_assess = len(test_groups[test_groups == 4])
    n_over_five_assess = len(test_groups[test_groups >= 5])

    index = np.arange(len(groups))

    first_assess_idx = []

    idx_groups_map = {}
    groups_idx_map: Dict[str, List[int]] = {}

    groups_has_more_than_two = []
    groups_has_more_than_three = []
    groups_has_more_than_four = []
    groups_has_more_than_five = []
    for inst_id in np.unique(groups):
        idx_inst = index[groups == inst_id]
        groups_idx_map[inst_id] = idx_inst
        for idx in idx_inst:
            idx_groups_map[idx] = inst_id

        first_assess_idx.append(idx_inst[0])
        if len(idx_inst) > 1:
            groups_has_more_than_two.append(inst_id)
        if len(idx_inst) > 2:
            groups_has_more_than_three.append(inst_id)
        if len(idx_inst) > 3:
            groups_has_more_than_four.append(inst_id)
        if len(idx_inst) > 4:
            groups_has_more_than_five.append(inst_id)

    first_assess_dice = []
    used_groups_list = []
    for i in range(n_trials):
        dice = np.random.choice(
            first_assess_idx, size=n_first_assess, replace=False)
        first_assess_dice.append(dice)
        used_groups = set(map(lambda x: idx_groups_map[x], dice))
        used_groups_list.append(used_groups)

    first_assess = np.asarray(first_assess_dice).T

    second_assess_dice = []
    for i in range(n_trials):
        second_assess_idx = []
        used_groups = used_groups_list[i]
        available_groups = set(groups_has_more_than_two) - used_groups
        for group in available_groups:
            second_assess_idx.append(groups_idx_map[group][1])
        dice = np.random.choice(
            second_assess_idx, size=n_second_assess, replace=False)
        second_assess_dice.append(dice)
        used_groups_second = set(map(lambda x: idx_groups_map[x], dice))
        used_groups_list[i] = used_groups.union(used_groups_second)

    second_assess = np.asarray(second_assess_dice).T

    third_assess_dice = []
    for i in range(n_trials):
        third_assess_idx = []
        used_groups = used_groups_list[i]
        available_groups = set(groups_has_more_than_three) - used_groups
        for group in available_groups:
            third_assess_idx.append(groups_idx_map[group][2])
        dice = np.random.choice(
            third_assess_idx, size=n_third_assess, replace=False)
        third_assess_dice.append(dice)
        used_groups_third = set(map(lambda x: idx_groups_map[x], dice))
        used_groups_list[i] = used_groups.union(used_groups_third)

    third_assess = np.asarray(third_assess_dice).T
    fourth_asses_dice = []
    for i in range(n_trials):
        fourth_assess_idx = []
        used_groups = used_groups_list[i]
        available_groups = set(groups_has_more_than_four) - used_groups
        for group in available_groups:
            fourth_assess_idx.append(groups_idx_map[group][3])
        dice = np.random.choice(
            fourth_assess_idx, size=n_fourth_assess, replace=False)
        fourth_asses_dice.append(dice)
        used_groups_fourth = set(map(lambda x: idx_groups_map[x], dice))
        used_groups_list[i] = used_groups.union(used_groups_fourth)

    fourth_assess = np.asarray(fourth_asses_dice).T

    over_fifth_assess_dice = []
    for i in range(n_trials):
        over_fifth_assess_idx = []
        used_groups = used_groups_list[i]
        available_groups = set(groups_has_more_than_five) - used_groups
        for group in available_groups:
            over_fifth_assess_idx.extend(groups_idx_map[group][4:])
        dice = np.random.choice(
            over_fifth_assess_idx, size=n_over_five_assess, replace=False)
        over_fifth_assess_dice.append(dice)

    over_fifth_assess = np.asarray(over_fifth_assess_dice).T

    assess = np.vstack([
        first_assess, second_assess, third_assess, fourth_assess,
        over_fifth_assess
    ])

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
