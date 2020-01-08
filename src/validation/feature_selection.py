import numpy as np
import pandas as pd

from typing import List

from tqdm import tqdm


def select_features(cols: List[str],
                    feature_importance: pd.DataFrame,
                    config: dict,
                    delete_higher_importance: bool = False) -> List[str]:
    if config["val"].get("n_delete") is None:
        return cols

    n_delete = config["val"].get("n_delete")
    importance_sorted_cols = feature_importance.sort_values(
        by="value",
        ascending=not (delete_higher_importance))["feature"].tolist()
    if isinstance(n_delete, int):
        remove_cols = importance_sorted_cols[:n_delete]
        cols = [col for col in cols if col not in remove_cols]
    elif isinstance(n_delete, float):
        n_delete_int = int(n_delete * len(importance_sorted_cols))
        remove_cols = importance_sorted_cols[:n_delete_int]
        cols = [col for col in cols if col not in remove_cols]
    return cols


def remove_correlated_features(df: pd.DataFrame, features: List[str]):
    counter = 0
    to_remove: List[str] = []
    for i in tqdm(range(len(features) - 1)):
        feat_a = features[i]
        for j in range(i + 1, len(features)):
            feat_b = features[j]
            if feat_a in to_remove or feat_b in to_remove:
                continue
            c = np.corrcoef(df[feat_a], df[feat_b])[0][1]
            if c > 0.995:
                counter += 1
                to_remove.append(feat_b)
                print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(
                    counter, feat_a, feat_b, c))
    return to_remove
