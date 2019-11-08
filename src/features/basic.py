import numpy as np
import pandas as pd

from typing import List, Union, Dict

from tqdm import tqdm

from src.features.base import Feature, PartialFeature

IoF = Union[int, float]


class Basic(Feature):
    def create_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        all_activities = set(train_df["title"].unique()).union(
            set(test_df["title"].unique()))
        all_event_codes = set(train_df["event_code"].unique()).union(
            test_df["event_code"].unique())
        activities_map = dict(
            zip(all_activities, np.arange(len(all_activities))))
        inverse_activities_map = dict(
            zip(np.arange(len(all_activities)), all_activities))

        compiled_data_train: List[List[IoF]] = []
        compiled_data_test: List[List[IoF]] = []

        installation_ids_train = []
        installation_ids_test = []

        for ins_id, user_sample in tqdm(
                train_df.groupby("installation_id", sort=False),
                total=train_df["installation_id"].nunique()):
            installation_ids_train.append(ins_id)
            feats = KernelFeatures(all_activities, all_event_codes,
                                   activities_map, inverse_activities_map)
            compiled_data_train.extend(
                feats.create_features(user_sample, test=False))

        for ins_id, user_sample in tqdm(
                test_df.groupby("installation_id", sort=False),
                total=test_df["installation_id"].nunique()):
            installation_ids_test.append(ins_id)
            feats = KernelFeatures(all_activities, all_event_codes,
                                   activities_map, inverse_activities_map)
            compiled_data_test.extend(
                feats.create_features(user_sample, test=True))


class KernelFeatures(PartialFeature):
    def __init__(self, all_activities: set, all_event_codes: set,
                 activities_map: Dict[str, float],
                 inverse_activities_map: Dict[float, str]):
        self.all_activities = all_activities
        self.all_event_codes = all_event_codes
        self.activities_map = activities_map
        self.inverse_activities_map = inverse_activities_map

        super().__init__()

    def create_features(self, df: pd.DataFrame, test: bool = False):
        time_spent_each_act = {act: 0 for act in self.all_activities}

        for i, sess in df.groupby("game_session", sort=False):
            sess_type = sess["type"].iloc[0]
            sess_title = sess["title"].iloc[0]

            if sess_type != "Assessment":
                time_spent = int(sess["game_time"].iloc[-1] / 1000)
                time_spent_each_act[
                    self.inverse_activities_map[sess_title]] += time_spent

            if sess_type == "Assessment" and (test or len(sess) > 1):
                raise NotImplementedError
