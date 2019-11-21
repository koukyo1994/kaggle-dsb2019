import numpy as np
import pandas as pd

from typing import List, Union, Dict, Optional, Tuple

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.features.base import Feature

IoF = Union[int, float]
IoS = Union[int, str]


class ImprovedBasic(Feature):
    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        train_df = train.copy()
        test_df = test.copy()

        all_event_codes = {
            2000,
            2020,
            2030,
            3010,
            3020,
            3021,
            3110,
            3120,
            3021,
            4010,
            4020,
            4025,
            4030,
            4035,
            4040,
            4070,
            4090,
            4100,
        }

        compiled_data_train: List[pd.DataFrame] = []
        compiled_data_valid: List[pd.DataFrame] = []
        compiled_data_test: List[pd.DataFrame] = []

        installation_ids_train = []
        installation_ids_valid = []
        installation_ids_test = []

        train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
        test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

        for ins_id, user_sample in tqdm(
                train_df.groupby("installation_id", sort=False),
                total=train_df["installation_id"].nunique(),
                desc="train basic features"):
            if "Assessment" not in user_sample["type"].unique():
                continue

            feat_df, _ = improved_basic_features(
                user_sample, all_event_codes, test=False)

            installation_ids_train.extend([ins_id] * len(feat_df))
            compiled_data_train.append(feat_df)
        le = LabelEncoder()
        self.train = pd.concat(compiled_data_train, axis=0, sort=False)
        self.train["session_title"] = le.fit_transform(
            self.train["session_title"])
        self.train["installation_id"] = installation_ids_train
        self.train.reset_index(drop=True, inplace=True)

        for ins_id, user_sample in tqdm(
                test_df.groupby("installation_id", sort=False),
                total=test_df["installation_id"].nunique(),
                desc="test features"):
            feat_df, valid_df = improved_basic_features(
                user_sample, all_event_codes, test=True)
            installation_ids_valid.extend(
                [ins_id] * len(valid_df))  # type: ignore
            installation_ids_test.extend([ins_id] * len(feat_df))
            compiled_data_valid.append(valid_df)
            compiled_data_test.append(feat_df)
        self.valid = pd.concat(compiled_data_valid, axis=0, sort=False)
        self.valid["session_title"] = le.transform(self.valid["session_title"])
        self.valid["installation_id"] = installation_ids_valid
        self.valid.reset_index(drop=True, inplace=True)

        self.test = pd.concat(compiled_data_test, axis=0, sort=False)
        self.test["session_title"] = le.transform(self.test["session_title"])
        self.test["installation_id"] = installation_ids_test
        self.test.reset_index(drop=True, inplace=True)


def improved_basic_features(user_sample: pd.DataFrame,
                            all_event_codes: set,
                            test: bool = False
                            ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    event_code_count = {ev: 0 for ev in all_event_codes}
    user_activities_count: Dict[IoS, IoF] = {
        "Clip": 0,
        "Activity": 0,
        "Assessment": 0,
        "Game": 0
    }

    accumulated_acc_groups = 0.0
    accumulated_acc = 0.0
    accumulated_correct_attempts = 0
    accumulated_failed_attempts = 0
    accumulated_actions = 0

    counter = 0

    accuracy_group: Dict[IoS, IoF] = {0: 0, 1: 0, 2: 0, 3: 0}
    durations: List[float] = []
    last_activity = ""
    all_assessments: List[Dict[IoS, IoF]] = []
    for sess_id, sess in user_sample.groupby("game_session", sort=False):
        sess_type = sess["type"].iloc[0]
        if sess_type != "Assessment":
            pass

        if sess_type == "Assessment" and (test or len(sess) > 1):
            sess_title = sess["title"].iloc[0]

            attempt_code = 4110 if (
                sess_title == "Bird Measurer (Assessment)") else 4100
            all_attempts: pd.DataFrame = sess.query(
                f"event_code == {attempt_code}")
            correct_attempt = all_attempts["event_data"].str.contains(
                "true").sum()
            failed_attempt = all_attempts["event_data"].str.contains(
                "false").sum()

            features = user_activities_count.copy()
            features.update(event_code_count.copy())

            features["session_title"] = sess_title
            features["accumulated_correct_attempts"] = \
                accumulated_correct_attempts
            features["accumulated_failed_attempts"] = \
                accumulated_failed_attempts

            accumulated_correct_attempts += correct_attempt
            accumulated_failed_attempts += failed_attempt

            features["duration_mean"] = np.mean(durations) if durations else 0
            durations.append((sess.iloc[-1, 2] - sess.iloc[0, 2]).seconds)

            features["accumulated_acc"] = \
                accumulated_acc / counter if counter > 0 else 0
            acc = correct_attempt / (correct_attempt + failed_attempt) \
                if (correct_attempt + failed_attempt) != 0 else 0
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
            accuracy_group[features["accuracy_group"]] += 1  # type: ignore

            features["accumulated_accuracy_group"] = \
                accumulated_acc_groups / counter if counter > 0 else 0
            accumulated_acc_groups += features["accuracy_group"]

            features["accumulated_actions"] = accumulated_actions

            if len(sess) == 1:
                all_assessments.append(features)
            elif correct_attempt + failed_attempt > 0:
                all_assessments.append(features)

            counter += 1

        num_event_codes: dict = sess["event_code"].value_counts().to_dict()
        for k in num_event_codes.keys():
            if k not in event_code_count.keys():
                continue
            event_code_count[k] += num_event_codes[k]

        accumulated_actions += len(sess)
        if last_activity != sess_type:
            user_activities_count[sess_type] += 1
            last_activity = sess_type

    if test:
        df = pd.DataFrame([all_assessments[-1]])
        valid_df = pd.DataFrame(all_assessments[:-1])
        return df, valid_df
    else:
        df = pd.DataFrame(all_assessments)
        return df, None
