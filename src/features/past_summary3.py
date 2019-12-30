import json

import numpy as np
import pandas as pd

from collections import Counter
from typing import Dict, List, Optional, Tuple, Union, Any

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.features.base import Feature
from src.features.modules import TargetEncoder

IoF = Union[int, float]
IoS = Union[int, str]


class PastSummary3(Feature):
    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        train_df = train.copy()
        test_df = test.copy()

        train_df["title_event_code"] = list(
            map(lambda x, y: str(x) + "_" + str(y), train["title"],
                train["event_code"]))
        test_df["title_event_code"] = list(
            map(lambda x, y: str(x) + "_" + str(y), test["title"],
                test["event_code"]))

        event_codes = [
            2000, 4070, 3120, 3121, 4020, 4035, 4030, 3110, 4022, 2010, 4090
        ]
        assessments = [
            "Mushroom Sorter (Assessment)", "Bird Measurer (Assessment)",
            "Cauldron Filler (Assessment)", "Cart Balancer (Assessment)",
            "Chest Sorter (Assessment)"
        ]
        title_event_codes = [
            "Crystal Caves - Level 1_2000", "Crystal Caves - Level 2_2000",
            "Crystal Caves - Level 3_2000",
            "Cauldron Filler (Assessment)_3020",
            "Sandcastle Builder (Activity)_4070",
            "Sandcastle Builder (Activity)_4020",
            "Bug Measurer (Activity)_4035", "Chow Time_4070",
            "Bug Measurer (Activity)_4070", "All Star Sorting_2025"
        ]

        for assessment in assessments:
            for code in ["4020", "4070"]:
                title_event_codes.append(assessment + "_" + code)

        event_ids = ["27253bdc"]

        dfs_train: List[pd.DataFrame] = []
        dfs_valid: List[pd.DataFrame] = []
        dfs_test: List[pd.DataFrame] = []

        inst_ids_train: List[str] = []
        inst_ids_valid: List[str] = []
        inst_ids_test: List[str] = []

        train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
        test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

        train_df = add_date_features(train_df)
        test_df = add_date_features(test_df)

        for inst_id, user_sample in tqdm(
                train_df.groupby("installation_id", sort=False),
                total=train_df["installation_id"].nunique(),
                desc="train features"):
            if "Assessment" not in user_sample["type"].unique():
                continue
            feats, _ = past_summary_features(
                user_sample,
                event_codes,
                event_ids,
                title_event_codes,
                test=False)
            inst_ids_train.extend([inst_id] * len(feats))
            dfs_train.append(feats)

        le, le_world = LabelEncoder(), LabelEncoder()
        self.train = pd.concat(dfs_train, axis=0, sort=False)
        self.train["session_title"] = le.fit_transform(
            self.train["session_title"])
        self.train["world"] = le_world.fit_transform(self.train["world"])
        self.train["installation_id"] = inst_ids_train
        self.train.reset_index(drop=True, inplace=True)

        for inst_id, user_sample in tqdm(
                test_df.groupby("installation_id", sort=False),
                total=test_df["installation_id"].nunique(),
                desc="test features"):
            feats, valid_feats = past_summary_features(
                user_sample,
                event_codes,
                event_ids,
                title_event_codes,
                test=True)

            inst_ids_valid.extend([inst_id] * len(valid_feats))  # type: ignore
            inst_ids_test.extend([inst_id] * len(feats))
            dfs_valid.append(valid_feats)
            dfs_test.append(feats)

        self.valid = pd.concat(dfs_valid, axis=0, sort=False)
        self.valid["session_title"] = le.transform(self.valid["session_title"])
        self.valid["world"] = le_world.transform(self.valid["world"])
        self.valid["installation_id"] = inst_ids_valid
        self.valid.reset_index(drop=True, inplace=True)

        self.test = pd.concat(dfs_test, axis=0, sort=False)
        self.test["session_title"] = le.transform(self.test["session_title"])
        self.test["world"] = le_world.transform(self.test["world"])
        self.test["installation_id"] = inst_ids_test
        self.test.reset_index(drop=True, inplace=True)

        # pseudo target
        te = TargetEncoder(n_splits=10, random_state=4222)
        self.train["mean_target"] = te.fit_transform(
            self.train, self.train["accuracy_group"], column="session_title")
        self.valid["mean_target"] = te.transform(self.valid)
        self.test["mean_target"] = te.transform(self.test)


def past_summary_features(
        user_sample: pd.DataFrame,
        event_codes: list,
        event_ids: list,
        title_event_code: list,
        test: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    raise NotImplementedError


def add_date_features(df: pd.DataFrame):
    df["date"] = df["timestamp"].dt.date
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["weekofyear"] = df["timetamp"].dt.weekofyear
    return df
