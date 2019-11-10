import pandas as pd

from typing import List, Union

from tqdm import tqdm

from src.features.base import Feature

IoF = Union[int, float]


class PastClip(Feature):
    def create_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        raise NotImplementedError
