import pandas as pd

from typing import List, Union

from tqdm import tqdm

from src.features.base import Feature

IoF = Union[int, float]


class PastClip(Feature):
    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        train_df = train.copy()
        test_df = test.copy()
        train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
        test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

        compiled_data_train: List[pd.DataFrame] = []
        compiled_data_test: List[pd.DataFrame] = []
        for ins_id, user_sample in tqdm(
                train_df.groupby("installatioin_id", sort=False),
                total=train_df["installation_id"].nunique(),
                desc="train past clip"):
            if "Assessment" not in user_sample["type"].unique():
                continue
            feat_df = past_clip_features(user_sample, test=False)
            compiled_data_train.append(feat_df)

        self.train = pd.concat(compiled_data_train, axis=0, sort=False)
        self.train.reset_index(drop=True, inplace=True)

        for ins_id, user_sample in tqdm(
                test_df.groupby("installation_id", sort=False),
                total=test_df["installation_id"].nunique(),
                desc="test past clip"):
            feat_df = past_clip_features(user_sample, test=True)
            compiled_data_test.append(feat_df)
        self.test = pd.concat(compiled_data_test, axis=0, sort=False)
        self.test.reset_index(drop=True, inplace=True)


def past_clip_features(user_sample: pd.DataFrame,
                       test: bool = False) -> pd.DataFrame:
    raise NotImplementedError
