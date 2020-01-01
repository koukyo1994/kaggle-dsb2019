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
    def create_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        train_df["title_event_code"] = list(
            map(lambda x, y: str(x) + "_" + str(y), train_df["title"],
                train_df["event_code"]))
        test_df["title_event_code"] = list(
            map(lambda x, y: str(x) + "_" + str(y), test_df["title"],
                test_df["event_code"]))

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
            "Bug Measurer (Activity)_4070", "All Star Sorting_2025",
            "Leaf Leader_4070"
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
    event_code_count = {ev: 0 for ev in event_codes}
    event_id_count = {ev: 0 for ev in event_ids}
    title_event_code_count = {ev: 0 for ev in title_event_code}

    activities_count: Dict[IoS, IoF] = {}

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

    assessments = [
        "Mushroom Sorter (Assessment)", "Bird Measurer (Assessment)",
        "Cauldron Filler (Assessment)", "Cart Balancer (Assessment)",
        "Chest Sorter (Assessment)"
    ]

    action_code = {
        "Mushroom Sorter (Assessment)": [4020],
        "Bird Measurer (Assessment)": [4025],
        "Cauldron Filler (Assessment)": [4020],
        "Cart Balancer (Assessment)": [4020],
        "Chest Sorter (Assessment)": [4020, 4025]
    }

    games = [
        "Air Show", "All Star Sorting", "Bubble Bath", "Chow Time",
        "Crystals Rule", "Dino Drink", "Dino Dive", "Happy Camel",
        "Leaf Leader", "Pan Balance", "Scrub-A-Dub"
    ]

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

    activities = [
        "Bottle Filler (Activity)", "Bug Measurer (Activity)",
        "Chicken Balancer (Activity)", "Egg Dropper (Activity)",
        "Fireworks (Activity)", "Flower Waterer (Activity)",
        "Sandcastle Builder (Activity)", "Watering Hole (Activity)"
    ]

    past_assess_summary: Dict[str, List[Dict[str, Any]]] = {
        title: []
        for title in assessments
    }
    last_assessment: Tuple[str, Dict[str, Any]] = ("", {})

    past_game_summarys: Dict[str, List[Dict[str, IoF]]] = {
        title: []
        for title in games
    }

    past_activity_summarys: Dict[str, List[Dict[str, IoF]]] = {
        title: []
        for title in activities
    }

    for sess_id, sess in user_sample.groupby("game_session", sort=False):
        sess_type = sess["type"].iloc[0]

        if sess_type == "Activity":
            act_title = sess["title"].iloc[0]

            summary = {}
            if act_title == "Sandcastle Builder (Activity)":
                event_data = get_event_data(sess)
                summary["duration"] = event_data["game_time"].max()
                if "filled" not in event_data.columns:
                    summary["filled_eq_True"] = 0
                    summary["filled_eq_False"] = 0
                else:
                    for v in [True, False]:
                        summary[f"filled_eq_{str(v)}"] = \
                            (event_data["filled"] == v).sum()
            elif act_title == "Fireworks (Activity)":
                event_data = get_event_data(sess)
                if "launched" not in event_data.columns:
                    summary["launched_eq_True"] = 0
                    summary["launched_eq_False"] = 0
                else:
                    for v in [True, False]:
                        summary[f"launched_eq_{str(v)}"] = \
                            (event_data["launched"] == v).sum()
            elif act_title == "Bug Measurer (Activity)":
                event_data = get_event_data(sess)
                summary["duration"] = event_data["game_time"].max()
            else:
                pass

            past_activity_summarys[act_title].append(summary)

        if sess_type == "Game":
            game_title = sess["title"].iloc[0]
            summary = {}
            event_data = get_event_data(sess)

            n_round = event_data[game_count_unit[game_title]].max()

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
            world = sess["world"].iloc[0]

            attempt_code = 4110 if (
                sess_title == "Bird Measurer (Assessment)") else 4100
            all_attempts: pd.DataFrame = sess.query(
                f"event_code == {attempt_code}")
            correct_attempt = all_attempts["event_data"].str.contains(
                "true").sum()
            failed_attempt = all_attempts["event_data"].str.contains(
                "false").sum()

            # Basic Features
            features = activities_count.copy()
            features.update(event_code_count.copy())
            features.update(event_id_count.copy())
            features.update(title_event_code_count.copy())

            features["session_title"] = sess_title
            features["world"] = world
            features["month"] = sess["month"].iloc[0]
            features["year"] = sess["year"].iloc[0]
            features["hour"] = sess["hour"].iloc[0]
            features["dayofweek"] = sess["dayofweek"].iloc[0]
            features["weekofyear"] = sess["weekofyear"].iloc[0]

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
            features["memory_decay_coeff_from_last_same_assess"] = \
                memory_decay_coeff_from_last_same_assess

            # work on the same assessments
            if len(past_assess_summary[sess_title]) > 0:
                same_assessments = past_assess_summary[sess_title]
                features["n_failure_same_assess"] = sum(
                    collect(same_assessments, "failed_attempts"))
                features["success_ratio_same_assess"] = np.mean(
                    collect(same_assessments, "success_ratio"))
                features["success_var_same_assess"] = np.var(
                    collect(same_assessments, "success_ratio"))
                features["mean_accuracy_group_same_assess"] = np.mean(
                    collect(same_assessments, "accuracy_group"))
                features["mean_timte_to_get_success_same_assess"] = np.mean(
                    collect(same_assessments, "time_to_get_success"))
                features["var_time_to_get_success_same_assess"] = np.var(
                    collect(same_assessments, "time_to_get_success"))
                features["mean_action_time_same_assess"] = np.mean(
                    collect(same_assessments, "mean_action_time"))
                features["var_action_time_same_assess"] = np.var(
                    collect(same_assessments, "mean_action_time"))
                features["mean_var_action_time_same_assess"] = np.mean(
                    collect(same_assessments, "var_action_time"))

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
                features["var_action_time_last_same_assess"] = \
                    same_assessments[-1]["var_action_time"]
            else:
                features["n_failure_same_assess"] = -1.0
                features["success_ratio_same_assess"] = -1.0
                features["success_var_same_assess"] = -1.0
                features["mean_accuracy_group_same_assess"] = -1.0
                features["mean_timte_to_get_success_same_assess"] = -1.0
                features["var_time_to_get_success_same_assess"] = -1.0
                features["mean_action_time_same_assess"] = -1.0
                features["var_action_time_same_assess"] = -1.0
                features["mean_var_action_time_same_assess"] = -1.0
                features["n_failure_last_same_assess"] = -1.0
                features["success_ratio_last_same_assess"] = -1.0
                features["accuracy_group_last_same_assess"] = -1.0
                features["time_to_get_success_last_same_assess"] = -1.0
                features["mean_action_time_last_same_assess"] = -1.0
                features["var_action_time_last_same_assess"] = -1.0

            for assess in assessments:
                summs = past_assess_summary[assess]
                if len(summs) > 0:
                    features[assess + "_success_ratio"] = np.mean(
                        collect(summs, "success_ratio"))
                    features[assess + "_accuracy_group"] = np.mean(
                        collect(summs, "accuracy_group"))
                    features[assess + "_time_to_get_success"] = np.mean(
                        collect(summs, "time_to_get_success"))
                    features[assess + "_mean_action_time"] = np.mean(
                        collect(summs, "mean_action_time"))
                    features[assess + "_var_mean_action_time"] = np.var(
                        collect(summs, "mean_action_time"))
                    features[assess + "_mean_var_action_time"] = np.mean(
                        collect(summs, "var_action_time"))
                    features[assess + "_4070_mean"] = np.mean(
                        collect(summs, "4070"))
                    features[assess + "_4070_var"] = np.var(
                        collect(summs, "4070"))
                    if assess == "Cauldron Filler (Assessment)":
                        features[assess + "_3020_mean"] = np.mean(
                            collect(summs, "3020"))
                        features[assess + "_3020_var"] = np.var(
                            collect(summs, "3020"))
                else:
                    features[assess + "_success_raito"] = -1.0
                    features[assess + "_accuracy_group"] = -1.0
                    features[assess + "_time_to_get_success"] = -1.0
                    features[assess + "_mean_action_time"] = -1.0
                    features[assess + "_var_mean_action_time"] = -1.0
                    features[assess + "_mean_var_action_time"] = -1.0
                    features[assess + "_4070_mean"] = -1.0
                    features[assess + "_4070_var"] = -1.0
                    if assess == "Cauldron Filler (Assessment)":
                        features[assess + "_3020_mean"] = -1.0
                        features[assess + "_3020_var"] = -1.0

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
            summary_["4070"] = (sess["event_code"] == 4070).sum()
            summary_["3020"] = (sess["event_code"] == 3020).sum()

            action_diff = sess[sess["event_code"].isin(
                action_code[sess_title])]["timestamp"].diff().map(
                    lambda x: x.seconds).fillna(0).tolist()
            action_diff = list(filter(lambda x: x != 0.0, action_diff))
            summary_["mean_action_time"] = np.mean(action_diff)
            summary_["var_action_time"] = np.var(action_diff)
            if len(all_attempts) > 0:
                past_assess_summary[sess_title].append(summary_)
                last_assessment = (sess_title, summary_)

            # PastGame
            for game in games:
                features["n_last_correct_" + game] = 0.0
                features["mean_correct_" + game] = 0.0
                features["mean_incorrect_" + game] = 0.0
                features["var_correct" + game] = 0.0
                features["var_incorrect_" + game] = 0.0
                features["n_incorrect_" + game] = 0.0
                features["n_last_incorrect_" + game] = 0.0
                features["success_ratio_" + game] = 0.0
                features["var_success_ratio_" + game] = 0.0
                features["last_success_ratio_" + game] = 0.0
                features["count_4070_" + game] = 0.0
                features["mean_4070_" + game] = 0.0
                features["var_4070_" + game] = 0.0
                features["mean_action_time_" + game] = 0.0
                features["var_action_time_" + game] = 0.0

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
                features["var_correct_" + game] = np.var(
                    collect(summ, "mean_correct"))
                features["var_incorrect_" + game] = np.var(
                    collect(summ, "mean_incorrect"))
                features["success_ratio_" + game] = np.mean(
                    collect(summ, "mean_success_ratio"))
                features["var_success_ratio_" + game] = np.var(
                    collect(summ, "mean_success_ratio"))
                features["last_success_ratio_" + game] = collect(
                    summ, "mean_success_ratio")[-1]
                features["count_4070_" + game] = sum(
                    collect(summ, "count_4070"))
                features["mean_4070_" + game] = np.mean(
                    collect(summ, "count_4070"))
                features["var_4070_" + game] = np.var(
                    collect(summ, "count_4070"))
                features["mean_action_time_" + game] = np.mean(
                    collect(summ, "mean_action_time"))
                features["var_action_time_" + game] = np.var(
                    collect(summ, "mean_action_time"))

            # PastActivity
            for key, summs in past_activity_summarys.items():
                if key == "Fireworks (Activity)":
                    if len(summs) == 0:
                        features["n_launched_True"] = 0
                        features["n_launched_False"] = 0
                        features["launched_ratio"] = 0.0
                    else:
                        features["n_launched_True"] = sum(
                            collect(summs, "launched_eq_True"))
                        features["n_launched_False"] = sum(
                            collect(summs, "launched_eq_False"))
                        total = features["n_launched_False"] + \
                            features["n_launched_True"]
                        features["launched_ratio"] = \
                            features["n_launched_True"] / total \
                            if total > 0 else 0
                elif key == "Sandcastle Builder (Activity)":
                    if len(summs) == 0:
                        features[key + "_duration"] = 0
                        features["sand_filled_ratio"] = 0.0
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
                elif key == "Bug Measurer (Activity)":
                    if len(summs) == 0:
                        features[key + "_duration"] = 0
                    else:
                        features[key + "_duration"] = sum(
                            collect(summs, "duration"))

            if len(sess) == 1:
                all_assessments.append(features)
            elif correct_attempt + failed_attempt > 0:
                all_assessments.append(features)

            counter += 1

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
            last_activity = sess_type

    if test:
        df = pd.DataFrame([all_assessments[-1]])
        valid_df = pd.DataFrame(all_assessments[:-1])
        return df, valid_df
    else:
        df = pd.DataFrame(all_assessments)
        return df, None

    raise NotImplementedError


def add_date_features(df: pd.DataFrame):
    df["date"] = df["timestamp"].dt.date
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["weekofyear"] = df["timestamp"].dt.weekofyear
    return df


def get_event_data(df: pd.DataFrame) -> pd.DataFrame:
    return pd.io.json.json_normalize(df.event_data.apply(json.loads))


def collect(lod: List[Dict[str, IoF]], name: str) -> List[IoF]:
    return [d[name] for d in lod]
