import pandas as pd

from src.features.base import Feature


class Basic(Feature):
    def create_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        raise NotImplementedError
