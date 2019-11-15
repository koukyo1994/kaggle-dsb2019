import numpy as np
import pandas as pd

from typing import List, Union, Dict, Optional, Tuple

from tqdm import tqdm

from src.features.base import Feature, PartialFeature

IoF = Union[int, float]
IoS = Union[int, str]


class Basic(Feature):
    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        train_df = train.copy()
        test_df = test.copy()
        all_activities = set(train_df["title"].unique()).union(
            set(test_df["title"].unique()))
        all_event_codes = set(train_df["event_code"].unique()).union(
            test_df["event_code"].unique())
        activities_map = dict(
            zip(all_activities, np.arange(len(all_activities))))
        inverse_activities_map = dict(
            zip(np.arange(len(all_activities)), all_activities))

        compiled_data_train: List[pd.DataFrame] = []
        compiled_data_valid: List[pd.DataFrame] = []
        compiled_data_test: List[pd.DataFrame] = []

        installation_ids_train = []
        installation_ids_valid = []
        installation_ids_test = []

        train_df["title"] = train_df["title"].map(activities_map)
        test_df["title"] = test_df["title"].map(activities_map)

        train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
        test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

        for ins_id, user_sample in tqdm(
                train_df.groupby("installation_id", sort=False),
                total=train_df["installation_id"].nunique(),
                desc="train features"):
            if "Assessment" not in user_sample["type"].unique():
                continue
            feats = KernelFeatures(all_activities, all_event_codes,
                                   activities_map, inverse_activities_map)
            feat_df, _ = feats.create_features(user_sample, test=False)

            installation_ids_train.extend([ins_id] * len(feat_df))
            compiled_data_train.append(feat_df)
        self.train = pd.concat(compiled_data_train, axis=0, sort=False)
        self.train["installation_id"] = installation_ids_train
        self.train.reset_index(drop=True, inplace=True)

        for ins_id, user_sample in tqdm(
                test_df.groupby("installation_id", sort=False),
                total=test_df["installation_id"].nunique(),
                desc="test features"):
            feats = KernelFeatures(all_activities, all_event_codes,
                                   activities_map, inverse_activities_map)
            feat_df, valid_df = feats.create_features(user_sample, test=True)
            installation_ids_valid.extend(
                [ins_id] * len(valid_df))  # type: ignore
            installation_ids_test.extend([ins_id] * len(feat_df))
            compiled_data_valid.append(valid_df)
            compiled_data_test.append(feat_df)
        self.valid = pd.concat(compiled_data_valid, axis=0, sort=False)
        self.valid["installation_id"] = installation_ids_valid
        self.valid.reset_index(drop=True, inplace=True)

        self.test = pd.concat(compiled_data_test, axis=0, sort=False)
        self.test["installation_id"] = installation_ids_test
        self.test.reset_index(drop=True, inplace=True)

        # for df in [self.train, self.valid, self.test]:
        #     df["installation_session_count"] = df.groupby(
        #         "installation_id")["Clip"].transform("count")
        #     df["installation_duration_mean"] = df.groupby(
        #         "installation_id")["duration_mean"].transform("mean")
        #     df["installation_title_nunique"] = df.groupby(
        #         "installation_id")["session_title"].transform("nunique")
        #     df['sum_event_code_count'] = df[[
        #         2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075,
        #         2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 4022,
        #         4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000,
        #         4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 2040, 4090,
        #         4220, 4095
        #     ]].sum(axis=1)
        #     df['installation_event_code_count_mean'] = df.groupby(
        #         ['installation_id'])['sum_event_code_count'].transform('mean')


class KernelFeatures(PartialFeature):
    def __init__(self, all_activities: set, all_event_codes: set,
                 activities_map: Dict[str, float],
                 inverse_activities_map: Dict[float, str]):
        self.all_activities = all_activities
        self.all_event_codes = all_event_codes
        self.activities_map = activities_map
        self.inverse_activities_map = inverse_activities_map

        win_code = dict(
            zip(activities_map.values(),
                (4100 * np.ones(len(activities_map))).astype(int)))
        win_code[activities_map["Bird Measurer (Assessment)"]] = 4110
        self.win_code = win_code

        super().__init__()

    def create_features(self, df: pd.DataFrame, test: bool = False
                        ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        time_spent_each_act = {act: 0 for act in self.all_activities}
        event_code_count = {ev: 0 for ev in self.all_event_codes}
        user_activities_count: Dict[IoS, IoF] = {
            "Clip": 0,
            "Activity": 0,
            "Assessment": 0,
            "Game": 0
        }

        all_assesments = []

        accumulated_acc_groups = 0
        accumulated_acc = 0
        accumulated_correct_attempts = 0
        accumulated_failed_attempts = 0
        accumulated_actions = 0

        counter = 0

        accuracy_group: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

        durations: List[float] = []
        last_activity = ""

        for i, sess in df.groupby("game_session", sort=False):
            sess_type = sess["type"].iloc[0]
            sess_title = sess["title"].iloc[0]

            if sess_type != "Assessment":
                time_spent = int(sess["game_time"].iloc[-1] / 1000)
                time_spent_each_act[
                    self.inverse_activities_map[sess_title]] += time_spent

            if sess_type == "Assessment" and (test or len(sess) > 1):
                all_attempts: pd.DataFrame = sess.query(
                    f"event_code == {self.win_code[sess_title]}")
                true_attempt = all_attempts["event_data"].str.contains(
                    "true").sum()
                false_attempt = all_attempts["event_data"].str.contains(
                    "false").sum()

                features = user_activities_count.copy()
                features.update(time_spent_each_act.copy())
                features.update(event_code_count.copy())

                features["session_title"] = sess_title

                features["accumulated_correct_attempts"] = \
                    accumulated_correct_attempts
                features["accumulated_failed_attempts"] = \
                    accumulated_failed_attempts

                accumulated_correct_attempts += true_attempt
                accumulated_failed_attempts += false_attempt

                features["duration_mean"] = np.mean(
                    durations) if durations else 0
                durations.append((sess.iloc[-1, 2] - sess.iloc[0, 2]).seconds)

                features["accumulated_acc"] = \
                    accumulated_acc / counter if counter > 0 else 0

                acc = true_attempt / (true_attempt + false_attempt) \
                    if (true_attempt + false_attempt) != 0 else 0
                accumulated_acc += acc

                if acc == 0:
                    features["accuracy_group"] = 0
                elif acc == 1:
                    features["accuracy_group"] = 3
                elif acc == 0.5:
                    features["accuracy_group"] = 2
                else:
                    features["accuracy_group"] = 1

                features.update(accuracy_group.copy())
                accuracy_group[features["accuracy_group"]] += 1

                features["accumulated_accuracy_group"] = \
                    accumulated_acc_groups / counter if counter > 0 else 0
                accumulated_acc_groups += features["accuracy_group"]

                features["accumulated_actions"] = accumulated_actions

                if len(sess) == 1:
                    all_assesments.append(features)
                elif true_attempt + false_attempt > 0:
                    all_assesments.append(features)

                counter += 1

            num_event_codes: dict = sess["event_code"].value_counts().to_dict()
            for k in num_event_codes.keys():
                event_code_count[k] += num_event_codes[k]

            accumulated_actions += len(sess)
            if last_activity != sess_type:
                user_activities_count[sess_type] + +1
                last_activity = sess_type

        if test:
            df = pd.DataFrame([all_assesments[-1]])
            valid_df = pd.DataFrame(all_assesments[:-1])
            return df, valid_df
        else:
            df = pd.DataFrame(all_assesments)
            return df, None
