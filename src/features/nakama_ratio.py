import pandas as pd

from pathlib import Path

from .base import Feature


class Ratio(Feature):
    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        input_dir = Path("features")
        nakama_train = pd.read_feather(input_dir / "NakamaV8_train.ftr")
        nakama_valid = pd.read_feather(input_dir / "NakamaV8_valid.ftr")
        nakama_test = pd.read_feather(input_dir / "NakamaV8_test.ftr")

        cols = nakama_train.columns
        ratio_cols = [col for col in cols if "Ratio" in col]

        self.train = nakama_train[ratio_cols]
        self.valid = nakama_valid[ratio_cols]
        self.test = nakama_test[ratio_cols]
