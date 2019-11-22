import json

import numpy as np
import pandas as pd

from collections import Counter
from typing import Dict, List, Optional, Tuple, Union, Any

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.features.base import Feature

IoF = Union[int, float]
IoS = Union[int, str]


class PastSummary(Feature):
    def create_features(self, train: pd.DataFrame, test: pd.DataFrame):
        train_df = train.copy()
        test_df = test.copy()

        train_df['title_event_code'] = list(
            map(lambda x, y: str(x) + '_' + str(y), train['title'],
                train['event_code']))
        test_df['title_event_code'] = list(
            map(lambda x, y: str(x) + '_' + str(y), test['title'],
                test['event_code']))

        all_event_codes = {
            2000, 2020, 2030, 3010, 3020, 3021, 3110, 3120, 3021, 4010, 4020,
            4025, 4030, 4035, 4040, 4070, 4090, 4100
        }
        all_event_id = set(train_df["event_id"].unique()).union(
            set(test_df["event_id"].unique()))
        all_title_event_code = set(
            train_df["title_event_code"].unique()).union(
                set(test_df["title_event_code"].unique()))

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
                user_sample,
                all_event_codes,
                all_event_id,
                all_title_event_code,
                test=False)
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
                user_sample,
                all_event_codes,
                all_event_id,
                all_title_event_code,
                test=True)

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


def past_summary_features(
        user_sample: pd.DataFrame,
        all_event_codes: set,
        all_event_id: set,
        all_title_event_code: set,
        test: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    event_code_count = {ev: 0 for ev in all_event_codes}
    event_id_count = {ev: 0 for ev in all_event_id}
    title_event_code_count = {ev: 0 for ev in all_title_event_code}

    user_activities_count: Dict[IoS, IoF] = {
        "Clip": 0,
        "Activity": 0,
        "Assessment": 0,
        "Game": 0
    }

    all_assessments: List[Dict[IoS, IoF]] = []

    accumulated_acc_groups = 0.0
    accumulated_acc = 0.0
    accumulated_correct_attempts = 0
    accumulated_failed_attempts = 0
    accumulated_actions = 0

    counter = 0

    accuracy_group: Dict[IoS, IoF] = {0: 0, 1: 0, 2: 0, 3: 0}

    durations: List[float] = []
    last_activity = ""

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

    all_game_titles = {
        "Air Show", "All Star Sorting", "Bubble Bath", "Chow Time",
        "Crystals Rule", "Dino Dive", "Dino Drink", "Happy Camel",
        "Leaf Leader", "Pan Balance", "Scrub-A-Dub"
    }

    game_count_unit = {
        "All Star Sorting": "round",
        "Scrub-A-Dub": "level",
        "Air Show": "round",
        "Crystals Rule": "round",
        "Dino Drink": "round",
        "Bubble Bath": "round",
        "Dino Dive": "round",
        "Chow Time": "round",
        "Pan Balance": "round",
        "Happy Camel": "round",
        "Leaf Leader": "round"
    }

    activity_codes_map = {
        "Bottle Filler (Activity)": [4030, 4020, 4035, 4070],
        "Bug Measurer (Activity)": [4030, 4035, 4070],
        "Chicken Balancer (Activity)": [4070, 4030, 4020],
        "Egg Dropper (Activity)": [4020],
        "Fireworks (Activity)": [4030, 4020, 4070],
        "Flower Waterer (Activity)": [4070, 4030, 4020],
        "Sandcastle Builder (Activity)": [4070, 4030, 4035, 4020],
        "Watering Hole (Activity)": [4021, 4070]
    }

    activity_columns_map: Dict[
        str, Dict[str, Union[str, List[Union[str, bool]]]]] = {
            "Bottle Filler (Activity)": {
                "round": "max",
                "identifier": [],
                "jar_filled": [True, False]
            },
            "Bug Measurer (Activity)": {
                "identifier": ["sid_bugtank_line22", "sid_bugtank_line21"]
            },
            "Chicken Balancer (Activity)": {
                "identifier": [],
                "layout.left.pig": [False],
                "layout.right.pig": [False]
            },
            "Egg Dropper (Activity)": {
                "identifier": ["Buddy_EggsWentToOtherNest"]
            },
            "Fireworks (Activity)": {
                "identifier": ["Dot_SoHigh"],
                "launched": [True, False]
            },
            "Flower Waterer (Activity)": {
                "identifier": []
            },
            "Sandcastle Builder (Activity)": {
                "identifier": ["Dot_DragShovel", "Dot_SoCool", "Dot_FillItUp"],
                "filled": [True, False]
            },
            "Watering Hole (Activity)": {
                "identifier": [],
                "filled": [True, False]
            }
        }

    past_assess_summary: Dict[str, List[Dict[str, Any]]] = {
        title: []
        for title in assess_codes_map.keys()
    }
    last_assessment: Tuple[str, Dict[str, Any]] = ("", {})

    past_game_summarys: Dict[str, List[Dict[str, IoF]]] = {
        title: []
        for title in all_game_titles
    }
    past_activity_summarys: Dict[str, List[Dict[str, IoF]]] = {
        title: []
        for title in activity_codes_map.keys()
    }
    for sess_id, sess in user_sample.groupby("game_session", sort=False):
        sess_type = sess["type"].iloc[0]
        if ((sess_type != "Assessment") and (sess_type != "Game")):
            pass

        if sess["type"].iloc[0] == "Activity":
            act_title = sess["title"].iloc[0]
            event_data = get_event_data(sess)

            columns = activity_columns_map[act_title]
            codes = activity_codes_map[act_title]

            summary = {}
            for code in codes:
                summary[str(code)] = (sess["event_code"] == code).sum()

            summary["duration"] = event_data["game_time"].max()
            for key, val in columns.items():
                if val == "max":
                    summary["max_" + key] = max(event_data[key])
                if isinstance(val, list):
                    for v in val:
                        if key not in event_data.columns:
                            summary[key + "_eq_" + str(v)] = 0
                        else:
                            summary[key + "_eq_" + str(v)] = (
                                event_data[key] == v).sum()
            past_activity_summarys[act_title].append(summary)

        if sess_type == "Game":
            game_title = sess["title"].iloc[0]
            event_data = pd.io.json.json_normalize(sess["event_data"].apply(
                json.loads))
            summary = {}

            n_round = event_data[game_count_unit[game_title]].max()
            summary["n_max_round"] = n_round

            n_correct = len(event_data.query("event_code == 3021"))
            n_incorrect = len(event_data.query("event_code == 3020"))
            summary["n_correct"] = n_correct
            summary["n_incorrect"] = n_incorrect

            if n_round > 0:
                summary["mean_correct"] = n_correct / n_round
                summary["mean_incorrect"] = n_incorrect / n_round
            else:
                summary["mean_correct"] = 0
                summary["mean_incorrect"] = 0

            if (n_correct + n_incorrect) > 0:
                summary["mean_success_ratio"] = n_correct / (
                    n_correct + n_incorrect)
            else:
                summary["mean_success_ratio"] = 0.0
            summary["count_4070"] = (sess["event_code"] == 4070).sum()

            action_diff = sess[sess["event_code"].isin(
                [4020, 4025])]["timestamp"].diff().map(
                    lambda x: x.seconds).fillna(0).tolist()
            action_diff = list(filter(lambda x: x != 0.0, action_diff))
            summary["mean_action_time"] = np.mean(action_diff)
            past_game_summarys[game_title].append(summary)

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

            # Basic Features
            features = user_activities_count.copy()
            features.update(event_code_count.copy())
            features.update(event_id_count.copy())
            features.update(title_event_code_count.copy())

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

            # PastAssessment
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

            summary_: Dict[str, Any] = {}
            summary_["timestamp"] = sess["timestamp"].iloc[-1]
            summary_["n_attempts"] = len(all_attempts)
            if len(all_attempts) == 0:
                success_attempts = -1
                failed_attempts = -1
                success_ratio = -1.0
                accuracy_group_ = -1
                time_to_get_success = -1
            else:
                success_attempts = all_attempts["event_data"].str.contains(
                    "true").sum()
                failed_attempts = all_attempts["event_data"].str.contains(
                    "false").sum()
                success_ratio = success_attempts / len(all_attempts)
                if success_ratio == 0:
                    accuracy_group_ = 0
                elif success_ratio == 1:
                    accuracy_group_ = 3
                elif success_ratio == 0.5:
                    accuracy_group_ = 2
                else:
                    accuracy_group_ = 1

                if success_attempts > 0:
                    successed_att = all_attempts[all_attempts["event_data"].
                                                 str.contains("true")]
                    duration = (successed_att["timestamp"].iloc[0] -
                                sess["timestamp"].iloc[0]).seconds
                    time_to_get_success = duration
                else:
                    time_to_get_success = -1

            summary_["success_attempts"] = success_attempts
            summary_["failed_attempts"] = failed_attempts
            summary_["success_ratio"] = success_ratio
            summary_["accuracy_group"] = accuracy_group_
            summary_["time_to_get_success"] = time_to_get_success
            codes = assess_codes_map[sess_title]
            for code in codes:
                summary_[str(code)] = (sess["event_code"] == code).sum()
            action_diff = sess[sess["event_code"].isin(
                action_code[sess_title])]["timestamp"].diff().map(
                    lambda x: x.seconds).fillna(0).tolist()
            action_diff = list(filter(lambda x: x != 0.0, action_diff))
            summary_["mean_action_time"] = np.mean(action_diff)
            if len(all_attempts) > 0:
                past_assess_summary[sess_title].append(summary_)
                last_assessment = (sess_title, summary_)
            # PastGame
            # initialization
            for game in all_game_titles:
                features["n_last_correct_" + game] = 0.0
                features["mean_correct_" + game] = 0.0
                features["mean_incorrect_" + game] = 0.0
                features["n_incorrect_" + game] = 0.0
                features["n_last_incorrect_" + game] = 0.0
                features["success_ratio_" + game] = 0.0
                features["last_success_ratio_" + game] = 0.0
                features["count_4070_" + game] = 0.0
                features["mean_action_time_" + game] = 0.0

            for game, summ in past_game_summarys.items():
                if len(summ) == 0:
                    continue
                features["n_incorrect_" + game] = sum(
                    collect(summ, "n_incorrect"))
                features["n_last_correct_" + game] = collect(
                    summ, "n_correct")[-1]
                features["n_last_incorrect_" + game] = collect(
                    summ, "n_incorrect")[-1]
                features["mean_correct_" + game] = np.mean(
                    collect(summ, "mean_correct"))
                features["mean_incorrect_" + game] = np.mean(
                    collect(summ, "mean_incorrect"))
                features["success_ratio_" + game] = np.mean(
                    collect(summ, "mean_success_ratio"))
                features["last_success_ratio_" + game] = collect(
                    summ, "mean_success_ratio")[-1]
                features["count_4070_" + game] = sum(
                    collect(summ, "count_4070"))
                features["mean_action_time_" + game] = np.mean(
                    collect(summ, "mean_action_time"))

            for key, summs in past_activity_summarys.items():
                if key == "Bottle Filler (Activity)":
                    if len(summs) == 0:
                        features[key + "_duration"] = 0
                        features["mean_" + key + "_duration"] = 0
                        features["n_jar_filled_ratio"] = 0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = 0
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = 0
                    else:
                        max_round = max(collect(summs, "max_round"))
                        features[key + "_duration"] = sum(
                            collect(summs, "duration"))
                        features["mean_" + key + "_duration"] = \
                            features[key + "_duration"] / max_round
                        n_jar_filled_True = sum(
                            collect(summs, "jar_filled_eq_True"))
                        n_jar_filled_False = sum(
                            collect(summs, "jar_filled_eq_False"))
                        total = n_jar_filled_False + n_jar_filled_True
                        features["n_jar_filled_ratio"] = \
                            n_jar_filled_True / total \
                            if total > 0 else 0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = \
                                sum(collect(
                                    summs, "identifier" + "_eq_" + str(ident)))
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = sum(
                                collect(summs, str(code)))
                elif key == "Chicken Balancer (Activity)":
                    if len(summs) == 0:
                        features[key + "_duration"] = 0
                        features["n_layout.left.pig_False"] = 0
                        features["n_layout.right.pig_False"] = 0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_eq_" + str(ident) + "_count"] = 0
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = 0
                    else:
                        features[key + "_duration"] = sum(
                            collect(summs, "duration"))
                        features["n_layout.left.pig_False"] = sum(
                            collect(summs, "layout.left.pig_eq_False"))
                        features["n_layout.right.pig_False"] = sum(
                            collect(summs, "layout.right.pig_eq_False"))
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = \
                                sum(collect(
                                    summs, "identifier" + "_eq_" + str(ident)))
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = sum(
                                collect(summs, str(code)))
                elif key == "Fireworks (Activity)":
                    if len(summs) == 0:
                        features[key + "_duration"] = 0
                        features["n_launched_True"] = 0
                        features["n_launched_False"] = 0
                        features["launched_ratio"] = 0.0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = 0
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = 0
                    else:
                        features[key + "_duration"] = sum(
                            collect(summs, "duration"))
                        features["n_launched_True"] = sum(
                            collect(summs, "launched_eq_True"))
                        features["n_launched_False"] = sum(
                            collect(summs, "launched_eq_False"))
                        total = features["n_launched_False"] + \
                            features["n_launched_True"]
                        features["launched_ratio"] = \
                            features["n_launched_True"] / total \
                            if total > 0 else 0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = \
                                sum(collect(
                                    summs, "identifier" + "_eq_" + str(ident)))
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = sum(
                                collect(summs, str(code)))

                elif key == "Sandcastle Builder (Activity)":
                    if len(summs) == 0:
                        features[key + "_duration"] = 0
                        features["sand_filled_ratio"] = 0.0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = 0
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = 0
                    else:
                        features[key + "_duration"] = sum(
                            collect(summs, "duration"))
                        n_sand_filled_True = sum(
                            collect(summs, "filled_eq_True"))
                        n_sand_filled_False = sum(
                            collect(summs, "filled_eq_False"))
                        total = n_sand_filled_False + n_sand_filled_True
                        features["sand_filled_ratio"] = \
                            n_sand_filled_True / total \
                            if total > 0 else 0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = \
                                sum(collect(
                                    summs, "identifier" + "_eq_" + str(ident)))
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = sum(
                                collect(summs, str(code)))
                elif key == "Watering Hole (Activity)":
                    if len(summs) == 0:
                        features[key + "_duration"] = 0
                        features["water_filled_ratio"] = 0.0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = 0
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = 0
                    else:
                        features[key + "_duration"] = sum(
                            collect(summs, "duration"))
                        n_water_filled_True = sum(
                            collect(summs, "filled_eq_True"))
                        n_water_filled_False = sum(
                            collect(summs, "filled_eq_False"))
                        total = n_water_filled_False + n_water_filled_True
                        features["water_filled_ratio"] = \
                            n_water_filled_True / total \
                            if total > 0 else 0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = \
                                sum(collect(
                                    summs, "identifier" + "_eq_" + str(ident)))
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = sum(
                                collect(summs, str(code)))
                else:
                    if len(summs) == 0:
                        features[key + "_duration"] = 0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = 0
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = 0
                    else:
                        features[key + "_duration"] = sum(
                            collect(summs, "duration"))
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = \
                                sum(collect(
                                    summs, "identifier" + "_eq_" + str(ident)))
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = sum(
                                collect(summs, str(code)))

        def update_counters(counter: dict, col: str):
            num_of_session_count = Counter(sess[col])
            for k in num_of_session_count.keys():
                x = k
                if counter.get(k) is None:
                    continue
                counter[x] += num_of_session_count[k]
            return counter

        event_code_count = update_counters(event_code_count, "event_code")
        event_id_count = update_counters(event_id_count, "event_id")
        title_event_code_count = update_counters(title_event_code_count,
                                                 "title_event_code")

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


def collect(lod: List[Dict[str, IoF]], name: str) -> List[IoF]:
    return [d[name] for d in lod]


def get_event_data(df: pd.DataFrame) -> pd.DataFrame:
    return pd.io.json.json_normalize(df.event_data.apply(json.loads))
