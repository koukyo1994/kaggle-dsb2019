import numpy as np
import pandas as pd

from typing import List, Union, Dict, Tuple, Optional

from tqdm import tqdm

from src.features.base import Feature

IoF = Union[int, float]


class PastClip(Feature):
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
                desc="train past clip"):
            if "Assessment" not in user_sample["type"].unique():
                continue
            feat_df, _ = past_clip_features(user_sample, test=False)
            compiled_data_train.append(feat_df)

        self.train = pd.concat(compiled_data_train, axis=0, sort=False)
        self.train.reset_index(drop=True, inplace=True)

        for ins_id, user_sample in tqdm(
                test_df.groupby("installation_id", sort=False),
                total=test_df["installation_id"].nunique(),
                desc="test past clip"):
            feat_df, valid_df = past_clip_features(user_sample, test=True)
            compiled_data_valid.append(valid_df)
            compiled_data_test.append(feat_df)
        self.valid = pd.concat(compiled_data_valid, axis=0, sort=False)
        self.valid.reset_index(drop=True, inplace=True)
        self.test = pd.concat(compiled_data_test, axis=0, sort=False)
        self.test.reset_index(drop=True, inplace=True)


def past_clip_features(user_sample: pd.DataFrame, test: bool = False
                       ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
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
        "Bird Measurer (Assessment)": ["Bird Measurer", "Treasure Map"],
        "Cauldron Filler (Assessment)": ["Slop Problem"],
        "Cart Balancer (Assessment)":
        ["Balancing Act", "Honey Cake", "Lifting Heavy Things"],
        "Chest Sorter (Assessment)": ["Heavy, Heavier, Heaviest"]
    }
    all_assessments = []
    for sess_id, sess in user_sample.groupby("game_session", sort=False):
        if sess["type"].iloc[0] != "Assessment":
            continue

        if sess["type"].iloc[0] == "Assessment" and (test or len(sess) > 1):
            features: Dict[str, IoF] = {}
            # only Assessment session will go through here after
            sess_start_idx: int = sess.index.min()
            sess_title: str = sess.title.iloc[0]
            history: pd.DataFrame = user_sample.loc[:sess_start_idx - 1, :]

            relevant_clips = title_clip_relation[sess_title]
            clips_in_history = history.query("type == 'Clip'")
            watched_clips_names = clips_in_history["title"].unique()
            all_useful_clips = list(clip_length.keys())
            for clip in all_useful_clips:
                features[clip + "_completion"] = 0.0
                features["last_" + clip + "_completion"] = 0.0
                features["n_complete_" + clip] = 0.0
            features["avg_relevant_clips_completion"] = 0.0
            features["avg_n_complete_relevant_clips"] = 0.0
            features["last_relevant_clips_completion"] = 0.0

            relevant_clips_completion = []

            for clip in all_useful_clips:
                if clip in watched_clips_names:
                    past_clips = clips_in_history[clips_in_history["title"].
                                                  str.contains(clip)]
                    past_clips_idx = past_clips.index.values
                    completions = []
                    for idx in past_clips_idx:
                        if idx + 1 > history.index.max():
                            next_sess = sess.iloc[0, :]
                        else:
                            next_sess = history.loc[idx + 1, :]
                        pclip = past_clips.loc[idx, :]
                        if next_sess["installation_id"] != pclip[
                                "installation_id"]:
                            continue
                        duration = (
                            next_sess.timestamp - pclip.timestamp).seconds
                        completions.append(duration / clip_length[clip])
                    if len(completions) == 0:
                        continue
                    features[clip + "_completion"] = max(completions)
                    features["last_" + clip + "_completion"] = completions[-1]
                    features["n_complete_" + clip] = (np.array(completions) >
                                                      0.8).sum()
                    if clip in relevant_clips:
                        relevant_clips_completion.append(max(completions))
            if len(relevant_clips_completion) > 0:
                features["avg_relevant_clips_completion"] = np.mean(
                    relevant_clips_completion)
                features[
                    "last_relevant_clips_completion"] = relevant_clips_completion[
                        -1]
                features["avg_n_complete_relevant_clips"] = (
                    np.array(relevant_clips_completion) > 0.8).mean()

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
