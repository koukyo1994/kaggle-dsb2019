import json

import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Tuple, Union

from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.features.base import Feature

IoF = Union[int, float]
IoS = Union[int, str]


class RenewedFeatures(Feature):
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
            feats, _ = renewed_features(
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
            feats, valid_feats = renewed_features(
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


def renewed_features(user_sample: pd.DataFrame,
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

    clip_length = {
        "12 Monkeys": 120,
        "Balancing Act": 90,
        "Costume Box": 80,
        "Heavy, Heavier, Heaviest": 80,
        "Honey Cake": 160,
        "Lifting Heavy Things": 130,
        "Ordering Spheres": 75,
        "Pirate's Tale": 95,
        "Rulers": 140,
        "Slop Problem": 75,
        "Treasure Map": 170
    }

    title_clip_relation = {
        "Mushroom Sorter (Assessment)":
        ["12 Monkeys", "Costume Box", "Ordering Spheres", "Pirate's Tale"],
        "Bird Measurer (Assessment)": ["Rulers", "Treasure Map"],
        "Cauldron Filler (Assessment)": ["Slop Problem"],
        "Cart Balancer (Assessment)":
        ["Balancing Act", "Honey Cake", "Lifting Heavy Things"],
        "Chest Sorter (Assessment)": ["Heavy, Heavier, Heaviest"]
    }

    title_game_relation = {
        "Bird Measurer (Assessment)": ["Air Show", "Crystals Rule"],
        "Mushroom Sorter (Assessment)": ["All Star Sorting"],
        "Cauldron Filler (Assessment)":
        ["Bubble Bath", "Dino Dive", "Dino Drink", "Scrub-A-Dub"],
        "Cart Balancer (Assessment)": ["Chow Time", "Happy Camel"],
        "Chest Sorter (Assessment)": ["Leaf Leader", "Pan Balance"]
    }

    past_game_summarys: Dict[str, List[Dict[str, IoF]]] = {
        title: []
        for title in all_game_titles
    }

    past_clip_summarys: Dict[str, List[Dict[str, IoF]]] = {
        title: []
        for title in clip_length.keys()
    }

    for sess_id, sess in user_sample.groupby("game_session", sort=False):
        sess_type = sess["type"].iloc[0]
        if sess_type == "Activity":
            pass
        if sess_type == "Clip":
            clip_title = sess["title"].iloc[0]
            if clip_title in clip_length.keys():
                summary = {}
                sess_idx = sess.index.max()
                if sess_idx + 1 < user_sample.index.max():
                    next_sess = user_sample.loc[sess_idx + 1, :]
                    duration = (next_sess.timestamp -
                                sess.loc[sess_idx, "timestamp"]).seconds
                    summary["completion"] = duration / clip_length[clip_title]
                    summary["completed"] = int(
                        0.7 <= summary["completion"] < 1.5)
                    summary["too_long"] = int(summary["completion"] >= 1.5)
                    past_clip_summarys[clip_title].append(summary)

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
            sess_start_idx: int = sess.index.min()
            history: pd.DataFrame = user_sample.loc[:sess_start_idx - 1, :]
            features["has_same_assessment_in_past"] = \
                int(sess_title in history["title"].unique())
            features["n_same_assessment_in_past"] = \
                history.query(
                    f"title == '{sess_title}'")["game_session"].nunique()

            dur_from_last_assessment = np.iinfo(np.int).max
            if "Assessment" in history["type"].unique():
                delta = (sess.timestamp.min() -
                         history.query("type == 'Assessment'").timestamp.max())
                dur_from_last_assessment = delta.seconds
            features["memory_decay_coeff_from_last_assess"] = np.exp(
                -dur_from_last_assessment / 86400)

            dur_from_last_assessment = np.iinfo(np.int).max
            if features["has_same_assessment_in_past"] == 1:
                delta = (
                    sess.timestamp.min() -
                    history.query(f"title == '{sess_title}'").timestamp.max())
                dur_from_last_assessment = delta.seconds
            features["memory_decay_coeff_from_last_same_assess"] = np.exp(
                -dur_from_last_assessment / 86400)

            # work on the same assessments
            if features["has_same_assessment_in_past"] == 1:
                same_assessments = history.query(f"title == '{sess_title}'")
                attempt_code = 4110 if (
                    sess_title == "Bird Measurer (Assessment)") else 4100
                all_attempts = same_assessments.query(
                    f"event_code == {attempt_code}")
                succ_attempts = all_attempts["event_data"].str.contains(
                    "true").sum()
                fail_attempts = all_attempts["event_data"].str.contains(
                    "false").sum()
                features["n_success_same_assess"] = succ_attempts
                features["n_failure_same_assess"] = fail_attempts
                features["success_ratio_same_assess"] = succ_attempts / len(
                    all_attempts) if len(all_attempts) != 0 else 0

                # work on the last same assessment
                sess_id_last_same_assess = same_assessments[
                    "game_session"].unique()[-1]
                last_same_assess = history.query(
                    f"game_session == '{sess_id_last_same_assess}'")
                all_attempts = last_same_assess.query(
                    f"event_code == {attempt_code}")
                succ_attempts = all_attempts["event_data"].str.contains(
                    "true").sum()
                fail_attempts = all_attempts["event_data"].str.contains(
                    "false").sum()
                features["n_success_last_same_assess"] = succ_attempts
                features["n_failure_last_same_assess"] = fail_attempts
                acc = succ_attempts / len(all_attempts) \
                    if len(all_attempts) != 0 else 0
                features["success_ratio_last_same_assess"] = acc
                if acc == 0:
                    features["last_same_accuracy_group"] = 0
                elif acc == 1:
                    features["last_same_accuracy_group"] = 3
                elif acc == 0.5:
                    features["last_same_accuracy_group"] = 2
                else:
                    features["last_same_accuracy_group"] = 1
            else:
                features["n_success_same_assess"] = -1
                features["n_failure_same_assess"] = -1
                features["success_ratio_same_assess"] = -1
                features["n_success_last_same_assess"] = -1
                features["n_failure_last_same_assess"] = -1
                features["success_ratio_last_same_assess"] = -1
                features["last_same_accuracy_group"] = -1
            for col in [
                    "last_same_accuracy_group", "n_failure_last_same_assess",
                    "n_success_last_same_assess",
                    "success_ratio_last_same_assess"
            ]:
                features["decayed_" + col] = features[col] * features[
                    "memory_decay_coeff_from_last_same_assess"]
            # PastClip
            # initialization
            for clip in clip_length.keys():
                features["n_watched_" + clip] = 0.0
                features["n_completed_" + clip] = 0.0
                features["mean_completion_" + clip] = 0.0
                features["max_completion_" + clip] = 0.0
                features["sum_completion_" + clip] = 0.0
                features["n_too_much_" + clip] = 0.0
                features["last_completion_" + clip] = 0.0
                features["is_last_completed_" + clip] = 0.0
                features["is_last_overwatched_" + clip] = 0.0

            for clip, summ in past_clip_summarys.items():
                if len(summ) == 0:
                    continue
                features["n_watched_" + clip] = len(summ)
                features["n_completed_" + clip] = sum(
                    collect(summ, "completed"))
                features["mean_completion_" + clip] = np.mean(
                    collect(summ, "completion"))
                features["max_completion_" + clip] = max(
                    collect(summ, "completion"))
                features["sum_completion_" + clip] = sum(
                    collect(summ, "completion"))
                features["n_too_much_" + clip] = sum(collect(summ, "too_long"))
                features["last_completion_" + clip] = collect(
                    summ, "completion")[-1]
                features["is_last_completed_" + clip] = collect(
                    summ, "completed")[-1]
                features["is_last_overwatched_" + clip] = collect(
                    summ, "too_long")[-1]

            relevant_clips = title_clip_relation[sess_title]
            relevant_clip_stats = {
                "n_watched_relevant_clip": 0.0,
                "n_completed_relevant_clip": 0.0,
                "mean_completion_relevant_clip": 0.0,
                "max_completion_relevant_clip": 0.0,
                "sum_completion_relevant_clip": 0.0,
                "n_too_much_relevant_clip": 0.0,
                "mean_last_completion_relevant_clip": 0.0,
                "mean_is_last_completed_relevant_clip": 0.0,
                "mean_is_last_overwatched_relevant_clip": 0.0
            }
            mean_completions = []
            max_completions = []
            sum_completions = []
            last_completions = []
            is_last_completed_ = []
            is_last_overwatched_ = []
            for clip in relevant_clips:
                relevant_clip_stats["n_watched_relevant_clip"] = \
                    relevant_clip_stats["n_watched_relevant_clip"] + \
                    features[f"n_watched_{clip}"]
                relevant_clip_stats["n_completed_relevant_clip"] = \
                    relevant_clip_stats["n_completed_relevant_clip"] + \
                    features[f"n_completed_{clip}"]
                relevant_clip_stats["n_too_much_relevant_clip"] = \
                    relevant_clip_stats["n_too_much_relevant_clip"] + \
                    features[f"n_too_much_{clip}"]
                mean_completions.append(features[f"mean_completion_{clip}"])
                max_completions.append(features[f"max_completion_{clip}"])
                sum_completions.append(features[f"sum_completion_{clip}"])
                last_completions.append(features[f"last_completion_{clip}"])
                is_last_completed_.append(
                    features[f"is_last_completed_{clip}"])
                is_last_overwatched_.append(
                    features[f"is_last_overwatched_{clip}"])
            relevant_clip_stats["mean_completion_relevant_clip"] = np.mean(
                mean_completions)
            relevant_clip_stats["max_completion_relevant_clip"] = max(
                max_completions)
            relevant_clip_stats["sum_completion_relevant_clip"] = sum(
                sum_completions)
            relevant_clip_stats[
                "mean_last_completion_relevant_clip"] = np.mean(
                    last_completions)
            relevant_clip_stats[
                "mean_is_last_completed_relevant_clip"] = np.mean(
                    is_last_completed_)
            relevant_clip_stats[
                "mean_is_last_overwatched_relevant_clip"] = np.mean(
                    is_last_overwatched_)
            features.update(relevant_clip_stats)

            # PastGame
            # initialization
            for game in all_game_titles:
                features["n_trial_" + game] = 0.0
                features["n_max_round_" + game] = 0.0
                features["n_last_round_" + game] = 0.0
                features["n_correct_" + game] = 0.0
                features["n_last_correct_" + game] = 0.0
                features["mean_correct_" + game] = 0.0
                features["mean_incorrect_" + game] = 0.0
                features["n_incorrect_" + game] = 0.0
                features["n_last_incorrect_" + game] = 0.0
                features["success_ratio_" + game] = 0.0
                features["last_success_ratio_" + game] = 0.0

            for game, summ in past_game_summarys.items():
                if len(summ) == 0:
                    continue
                features["n_trial_" + game] = len(summ)
                features["n_max_round_" + game] = max(
                    collect(summ, "n_max_round"))
                features["n_last_round_" + game] = collect(
                    summ, "n_max_round")[-1]
                features["n_correct_" + game] = sum(collect(summ, "n_correct"))
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

            relevant_game_stats = {
                "n_trial_relevant_game": 0.0,
                "max_round_relevant_game": 0.0,
                "mean_last_round_relevant_game": 0.0,
                "n_correct_relevant_game": 0.0,
                "n_incorrect_relevant_game": 0.0,
                "n_last_correct_relevant_game": 0.0,
                "n_last_incorrect_relevant_game": 0.0,
                "mean_correct_relevant_game": 0.0,
                "mean_incorrect_relevant_game": 0.0,
                "success_ratio_relevant_game": 0.0,
                "last_success_ratio_relevant_game": 0.0
            }
            relevant_games = title_game_relation[sess_title]
            max_round = []
            last_round = []
            mean_correct = []
            mean_incorrect = []
            success_ratio = []
            last_success_ratio = []
            for game in relevant_games:
                relevant_game_stats["n_trial_relevant_game"] = \
                    relevant_game_stats["n_trial_relevant_game"] + \
                    features[f"n_trial_{game}"]
                max_round.append(features[f"n_max_round_{game}"])
                last_round.append(features[f"n_last_round_{game}"])
                relevant_game_stats["n_correct_relevant_game"] = \
                    relevant_game_stats["n_correct_relevant_game"] + \
                    features[f"n_correct_{game}"]
                relevant_game_stats["n_incorrect_relevant_game"] = \
                    relevant_game_stats["n_incorrect_relevant_game"] + \
                    features[f"n_incorrect_{game}"]
                relevant_game_stats["n_last_correct_relevant_game"] = \
                    relevant_game_stats["n_last_correct_relevant_game"] + \
                    features[f"n_last_correct_{game}"]
                relevant_game_stats["n_last_incorrect_relevant_game"] = \
                    relevant_game_stats["n_last_incorrect_relevant_game"] + \
                    features[f"n_last_incorrect_{game}"]
                mean_correct.append(features[f"mean_correct_{game}"])
                mean_incorrect.append(features[f"mean_incorrect_{game}"])
                success_ratio.append(features[f"success_ratio_{game}"])
                last_success_ratio.append(
                    features[f"last_success_ratio_{game}"])
            relevant_game_stats["max_round_relevant_game"] = max(max_round)
            relevant_game_stats["mean_last_round_relevant_game"] = np.mean(
                last_round)
            relevant_game_stats["mean_correct_relevant_game"] = np.mean(
                mean_correct)
            relevant_game_stats["mean_incorrect_relevant_game"] = np.mean(
                mean_incorrect)
            relevant_game_stats["success_ratio_relevant_game"] = np.mean(
                success_ratio)
            relevant_game_stats["last_success_ratio_relevant_game"] = np.mean(
                last_success_ratio)

            features.update(relevant_game_stats)

        num_event_codes: dict = sess["event_code"].value_counts().to_dict()
        for k in num_event_codes.keys():
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


def collect(lod: List[Dict[str, IoF]], name: str) -> List[IoF]:
    return [d[name] for d in lod]
