import json

import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Tuple, Union

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.features.base import Feature

IoF = Union[int, float]
IoS = Union[int, str]


class PastSummary(Feature):
    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        train_df = train.copy()
        test_df = test.copy()

        all_event_codes = set(train_df["event_code"].unique()).union(
            set(test_df["event_code"].unique()))

        dfs_train: List[pd.DataFrame] = []
        dfs_valid: List[pd.DataFrame] = []
        dfs_test: List[pd.DataFrame] = []

        inst_ids_train: List[str] = []
        inst_ids_valid: List[str] = []
        inst_ids_test: List[str] = []

        train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
        test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

        for inst_id, user_sample in tqdm(
                train_df.groupby("installation_id", sort=False),
                total=train_df["installation_id"].nunique(),
                desc="train features"):
            if "Assessment" not in user_sample["type"].unique():
                continue
            feats, _ = past_summary_features(
                user_sample, all_event_codes, test=False)
            inst_ids_train.extend([inst_id] * len(feats))
            dfs_train.append(feats)

        le = LabelEncoder()
        self.train = pd.concat(dfs_train, axis=0, sort=False)
        self.train["session_title"] = le.fit_transform(
            self.train["session_title"])
        self.train["installation_id"] = inst_ids_train
        self.train.reset_index(drop=True, inplace=True)

        for inst_id, user_sample in tqdm(
                test_df.groupby("installation_id", sort=False),
                total=test_df["installation_id"].nunique(),
                desc="test features"):
            feats, valid_feats = past_summary_features(
                user_sample, all_event_codes, test=True)

            inst_ids_valid.extend([inst_id] * len(valid_feats))  # type: ignore
            inst_ids_test.extend([inst_id] * len(feats))
            dfs_valid.append(valid_feats)
            dfs_test.append(feats)

        self.valid = pd.concat(dfs_valid, axis=0, sort=False)
        self.valid["session_title"] = le.transform(self.valid["session_title"])
        self.valid["installation_id"] = inst_ids_valid
        self.valid.reset_index(drop=True, inplace=True)

        self.test = pd.concat(dfs_test, axis=0, sort=False)
        self.test["session_title"] = le.transform(self.test["session_title"])
        self.test["installation_id"] = inst_ids_test
        self.test.reset_index(drop=True, inplace=True)


def past_summary_features(user_sample: pd.DataFrame,
                          all_event_codes: set,
                          test: bool = False
                          ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    raise NotImplementedError
