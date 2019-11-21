import numpy as np
import pandas as pd

from typing import Dict, List, Union, Tuple, Optional, Any

from tqdm import tqdm

from src.features.base import Feature

IoF = Union[int, float]


class ImprovedPastAssessment(Feature):
    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        train_df = train.copy()
        test_df = test.copy()
        train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
        test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

        compiled_data_train: List[pd.DataFrame] = []
        compiled_data_valid: List[pd.DataFrame] = []
        compiled_data_test: List[pd.DataFrame] = []

        for ins_id, user_sample in tqdm(
                train_df.groupby("installation_id", sort=False),
                total=train_df["installation_id"].nunique(),
                desc="train past assessment"):
            if "Assessment" not in user_sample["type"].unique():
                continue
            feat_df, _ = past_assess_features(user_sample, test=False)

            compiled_data_train.append(feat_df)
        self.train = pd.concat(compiled_data_train, axis=0, sort=False)
        self.train.reset_index(drop=True, inplace=True)

        for ins_id, user_sample in tqdm(
                test_df.groupby("installation_id", sort=False),
                total=test_df["installation_id"].nunique(),
                desc="test past assessment"):
            feat_df, valid_df = past_assess_features(user_sample, test=True)
            compiled_data_valid.append(valid_df)
            compiled_data_test.append(feat_df)
        self.valid = pd.concat(compiled_data_valid, axis=0, sort=False)
        self.valid.reset_index(drop=True, inplace=True)
        self.test = pd.concat(compiled_data_test, axis=0, sort=False)
        self.test.reset_index(drop=True, inplace=True)


def past_assess_features(user_sample: pd.DataFrame, test: bool = False
                         ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    assess_codes_map = {
        "Mushroom Sorter (Assessment)": [4070],
        "Bird Measurer (Assessment)": [],
        "Cauldron Filler (Assessment)": [4070, 3020],
        "Cart Balancer (Assessment)": [],
        "Chest Sorter (Assessment)": []
    }

    action_code = {
        "Mushroom Sorter (Assessment)": [4020],
        "Bird Measurer (Assessment)": [4025],
        "Cauldron Filler (Assessment)": [4020],
        "Cart Balancer (Assessment)": [4020],
        "Chest Sorter (Assessment)": [4020, 4025]
    }

    all_assessments = []
    past_assess_summary: Dict[str, List[Dict[str, Any]]] = {
        title: []
        for title in assess_codes_map.keys()
    }
    last_assessment: Tuple[str, Dict[str, Any]] = ("", {})
    for sess_id, sess in user_sample.groupby("game_session", sort=False):
        if sess["type"].iloc[0] != "Assessment":
            continue

        if sess["type"].iloc[0] == "Assessment" and (test or len(sess) > 1):
            features: Dict[str, IoF] = {}

            sess_title: str = sess.title.iloc[0]

            dur_from_last_assessment = np.iinfo(np.int).max
            if last_assessment[0] != "":
                delta = (
                    sess.timestamp.min() - last_assessment[1]["timestamp"])
                dur_from_last_assessment = delta.seconds
            features["memory_decay_coeff_from_last_assess"] = np.exp(
                -dur_from_last_assessment / 86400)

            dur_from_last_assessment = np.iinfo(np.int).max
            if len(past_assess_summary[sess_title]) > 0:
                last_same_assess = past_assess_summary[sess_title][-1]
                delta = (sess.timestamp.min() - last_same_assess["timestamp"])
                dur_from_last_assessment = delta.seconds
            memory_decay_coeff_from_last_same_assess = np.exp(
                -dur_from_last_assessment / 86400)

            # work on the same assessments
            if len(past_assess_summary[sess_title]) > 0:
                same_assessments = past_assess_summary[sess_title]
                features["n_failure_same_assess"] = sum(
                    collect(same_assessments, "failed_attempts"))
                features["success_ratio_same_assess"] = np.mean(
                    collect(same_assessments, "success_ratio"))
                features["mean_accuracy_group_same_assess"] = np.mean(
                    collect(same_assessments, "accuracy_group"))
                features["mean_time_to_get_success_same_assess"] = np.mean(
                    collect(same_assessments, "time_to_get_success"))
                features["mean_action_time_same_assess"] = np.mean(
                    collect(same_assessments, "mean_action_time"))

                # work on last same assess
                features["n_failure_last_same_assess"] = \
                    same_assessments[-1]["failed_attempts"]
                features["success_ratio_last_same_assess"] = \
                    same_assessments[-1]["success_ratio"]
                features["accuracy_group_last_same_assess"] = \
                    same_assessments[-1]["accuracy_group"]
                features["time_to_get_success_last_same_assess"] = \
                    same_assessments[-1]["time_to_get_success"]
                features["mean_action_time_last_same_assess"] = \
                    same_assessments[-1]["mean_action_time"]
            else:
                features["n_failure_same_assess"] = -1.0
                features["success_ratio_same_assess"] = -1.0
                features["mean_accuracy_group_same_assess"] = -1.0
                features["mean_time_to_get_success_same_assess"] = -1.0
                features["mean_action_time_same_assess"] = -1.0
                features["n_failure_last_same_assess"] = -1.0
                features["success_ratio_last_same_assess"] = -1.0
                features["accuracy_group_last_same_assess"] = -1.0
                features["time_to_get_success_last_same_assess"] = -1.0
                features["mean_action_time_last_same_assess"] = -1.0

            for key in assess_codes_map.keys():
                summs = past_assess_summary[key]
                if len(summs) > 0:
                    features[key + "_success_ratio"] = np.mean(
                        collect(summs, "success_ratio"))
                    features[key + "_accuracy_group"] = np.mean(
                        collect(summs, "accuracy_group"))
                    features[key + "_time_to_get_success"] = np.mean(
                        collect(summs, "time_to_get_success"))
                    features[key + "_mean_action_time"] = np.mean(
                        collect(summs, "mean_action_time"))
                    codes = assess_codes_map[key]
                    for code in codes:
                        features[key + f"_{str(code)}"] = sum(
                            collect(summs, str(code)))
                else:
                    features[key + "_success_ratio"] = -1.0
                    features[key + "_accuracy_group"] = -1.0
                    features[key + "_time_to_get_success"] = -1.0
                    features[key + "_mean_action_time"] = -1.0
                    codes = assess_codes_map[key]
                    for code in codes:
                        features[key + f"_{str(code)}"] = -1.0

                for col in [
                        "accuracy_group_last_same_assess",
                        "n_failure_last_same_assess",
                        "success_ratio_last_same_assess"
                ]:
                    features["decayed_" + col] = (
                        features[col] *
                        memory_decay_coeff_from_last_same_assess)

            attempt_code = 4110 if (
                sess_title == "Bird Measurer (Assessment)") else 4100
            all_attempts = sess.query(f"event_code == {attempt_code}")
            if len(sess) == 1:
                all_assessments.append(features)
            elif len(all_attempts) > 0:
                all_assessments.append(features)

            summary: Dict[str, Any] = {}
            summary["timestamp"] = sess["timestamp"].iloc[-1]
            summary["n_attempts"] = len(all_attempts)
            if len(all_attempts) == 0:
                success_attempts = -1
                failed_attempts = -1
                success_ratio = -1.0
                accuracy_group = -1
                time_to_get_success = -1
            else:
                success_attempts = all_attempts["event_data"].str.contains(
                    "true").sum()
                failed_attempts = all_attempts["event_data"].str.contains(
                    "false").sum()
                success_ratio = success_attempts / len(all_attempts)
                if success_ratio == 0:
                    accuracy_group = 0
                elif success_ratio == 1:
                    accuracy_group = 3
                elif success_ratio == 0.5:
                    accuracy_group = 2
                else:
                    accuracy_group = 1

                if success_attempts > 0:
                    successed_att = all_attempts[all_attempts["event_data"].
                                                 str.contains("true")]
                    duration = (successed_att["timestamp"].iloc[0] -
                                sess["timestamp"].iloc[0]).seconds
                    time_to_get_success = duration
                else:
                    time_to_get_success = -1

            summary["success_attempts"] = success_attempts
            summary["failed_attempts"] = failed_attempts
            summary["success_ratio"] = success_ratio
            summary["accuracy_group"] = accuracy_group
            summary["time_to_get_success"] = time_to_get_success
            codes = assess_codes_map[sess_title]
            for code in codes:
                summary[str(code)] = (sess["event_code"] == code).sum()
            action_diff = sess[sess["event_code"].isin(
                action_code[sess_title])]["timestamp"].diff().map(
                    lambda x: x.seconds).fillna(0).tolist()
            action_diff = list(filter(lambda x: x != 0.0, action_diff))
            summary["mean_action_time"] = np.mean(action_diff)
            if len(all_attempts) > 0:
                past_assess_summary[sess_title].append(summary)
                last_assessment = (sess_title, summary)

    if test:
        all_assess_df = pd.DataFrame([all_assessments[-1]])
        valid_df = pd.DataFrame(all_assessments[:-1])
        return all_assess_df, valid_df
    else:
        all_assess_df = pd.DataFrame(all_assessments)
        return all_assess_df, None


def collect(lod: List[Dict[str, Any]], name: str) -> List[Any]:
    return [d[name] for d in lod]
