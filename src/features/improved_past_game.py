import json

import numpy as np
import pandas as pd

from typing import List, Union, Dict, Tuple, Optional

from tqdm import tqdm

from src.features.base import Feature

IoF = Union[int, float]


class ImprovedPastGame(Feature):
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
                desc="train past game"):
            if "Assessment" not in user_sample["type"].unique():
                continue
            feat_df, _ = past_game_features(user_sample, test=False)
            compiled_data_train.append(feat_df)

        self.train = pd.concat(compiled_data_train, axis=0, sort=False)
        self.train.reset_index(drop=True, inplace=True)

        for ins_id, user_sample in tqdm(
                test_df.groupby("installation_id", sort=False),
                total=test_df["installation_id"].nunique(),
                desc="test past game"):
            feat_df, valid_df = past_game_features(user_sample, test=True)
            compiled_data_valid.append(valid_df)
            compiled_data_test.append(feat_df)
        self.valid = pd.concat(compiled_data_valid, axis=0, sort=False)
        self.valid.reset_index(drop=True, inplace=True)
        self.test = pd.concat(compiled_data_test, axis=0, sort=False)
        self.test.reset_index(drop=True, inplace=True)


def past_game_features(user_sample: pd.DataFrame, test: bool = False
                       ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
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

    all_assessments: List[Dict[str, IoF]] = []
    past_game_summarys: Dict[str, List[Dict[str, IoF]]] = {
        title: []
        for title in all_game_titles
    }
    for sess_id, sess in user_sample.groupby("game_session", sort=False):
        if ((sess["type"].iloc[0] != "Assessment")
                and (sess["type"].iloc[0] != "Game")):
            continue

        if sess["type"].iloc[0] == "Game":
            game_title = sess["title"].iloc[0]
            event_data = pd.io.json.json_normalize(sess["event_data"].apply(
                json.loads))
            past_game_summarys[game_title].append({
                "n_max_round": 0,
                "n_correct": 0,
                "n_incorrect": 0,
                "mean_correct": 0,
                "mean_incorrect": 0,
                "mean_success_ratio": 0.0,
                "count_4070": 0,
                "count_3020": 0,
                "count_3021": 0,
                "mean_action_time": 0.0
            })
            last_idx = len(past_game_summarys[game_title]) - 1
            n_round = event_data[game_count_unit[game_title]].max()
            past_game_summarys[game_title][last_idx]["n_max_round"] = \
                n_round
            n_correct = len(event_data.query("event_code == 3021"))
            n_incorrect = len(event_data.query("event_code == 3020"))
            past_game_summarys[game_title][last_idx]["n_correct"] = \
                n_correct
            past_game_summarys[game_title][last_idx]["n_incorrect"] = \
                n_incorrect
            if n_round > 0:
                past_game_summarys[game_title][last_idx]["mean_correct"] = \
                    n_correct / n_round
                past_game_summarys[game_title][last_idx]["mean_incorrect"] = \
                    n_incorrect / n_round
            if (n_correct + n_incorrect) > 0:
                past_game_summarys[game_title][
                    last_idx]["mean_success_ratio"] = \
                    (n_correct) / (n_correct + n_incorrect)
            past_game_summarys[game_title][last_idx]["count_4070"] = \
                (sess["event_code"] == 4070).sum()
            past_game_summarys[game_title][last_idx]["count_3020"] = \
                (sess["event_code"] == 3020).sum()
            past_game_summarys[game_title][last_idx]["count_3021"] = \
                (sess["event_code"] == 3021).sum()
            action_diff = sess[sess["event_code"].isin(
                [4020, 4025])]["timestamp"].diff().map(
                    lambda x: x.seconds).fillna(0).tolist()
            action_diff = list(filter(lambda x: x != 0.0, action_diff))
            past_game_summarys[game_title][last_idx][
                "mean_action_time"] = np.mean(action_diff)

        if sess["type"].iloc[0] == "Assessment" and (test or len(sess) > 1):
            features: Dict[str, IoF] = {}
            sess_title: str = sess.title.iloc[0]

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
                features["count_3020_" + game] = 0.0
                features["count_3021_" + game] = 0.0
                features["mean_action_time_" + game] = 0.0

            for game, summary in past_game_summarys.items():
                if len(summary) == 0:
                    continue
                features["n_incorrect_" + game] = sum(
                    collect(summary, "n_incorrect"))
                features["n_last_correct_" + game] = collect(
                    summary, "n_correct")[-1]
                features["n_last_incorrect_" + game] = collect(
                    summary, "n_incorrect")[-1]
                features["mean_correct_" + game] = np.mean(
                    collect(summary, "mean_correct"))
                features["mean_incorrect_" + game] = np.mean(
                    collect(summary, "mean_incorrect"))
                features["success_ratio_" + game] = np.mean(
                    collect(summary, "mean_success_ratio"))
                features["last_success_ratio_" + game] = collect(
                    summary, "mean_success_ratio")[-1]
                features["count_4070_" + game] = sum(
                    collect(summary, "count_4070"))
                features["count_3020_" + game] = sum(
                    collect(summary, "count_3020"))
                features["count_3021_" + game] = sum(
                    collect(summary, "count_3021"))
                features["mean_action_time_" + game] = np.mean(
                    collect(summary, "mean_action_time"))

            attempt_code = 4110 if (
                sess_title == "Bird Measurer (Assessment)") else 4100
            all_attempts = sess.query(f"event_code == {attempt_code}")
            if len(sess) == 1:
                all_assessments.append(features)
            elif len(all_attempts) > 0:
                all_assessments.append(features)

    if test:
        all_assess_df = pd.DataFrame([all_assessments[-1]])
        valid_df = pd.DataFrame(all_assessments[:-1])
        return all_assess_df, valid_df
    else:
        all_assess_df = pd.DataFrame(all_assessments)
        return all_assess_df, None


def collect(lod: List[Dict[str, IoF]], name: str) -> List[IoF]:
    return [d[name] for d in lod]
