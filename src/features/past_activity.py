import json

import pandas as pd

from typing import Dict, Union, Tuple, Optional, List

from tqdm import tqdm

from src.features.base import Feature

IoF = Union[int, float]


class PastActivity(Feature):
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
                desc="train past activity"):
            if "Assessment" not in user_sample["type"].unique():
                continue
            feat_df, _ = past_activity_features(user_sample, test=False)
            compiled_data_train.append(feat_df)

        self.train = pd.concat(compiled_data_train, axis=0, sort=False)
        self.train.reset_index(drop=True, inplace=True)

        for ins_id, user_sample in tqdm(
                test_df.groupby("installation_id", sort=False),
                total=test_df["installation_id"].nunique(),
                desc="test past activity"):
            feat_df, valid_df = past_activity_features(user_sample, test=True)
            compiled_data_valid.append(valid_df)
            compiled_data_test.append(feat_df)
        self.valid = pd.concat(compiled_data_valid, axis=0, sort=False)
        self.valid.reset_index(drop=True, inplace=True)
        self.test = pd.concat(compiled_data_test, axis=0, sort=False)
        self.test.reset_index(drop=True, inplace=True)


def past_activity_features(user_sample: pd.DataFrame, test: bool = False
                           ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    activity_codes_map = {
        "Bottle Filler (Activity)": [
            2020, 3010, 4030, 3110, 4020, 4035,
            2030, 4070, 2010],
        "Bug Measurer (Activity)": [
            3010, 4030, 3110, 4035, 4025, 4070
        ],
        "Chicken Balancer (Activity)": [
            3010, 3110, 4070, 4030, 4020, 4035, 4022
        ],
        "Egg Dropper (Activity)": [
            3010, 2020, 3110, 4025, 4020, 4070
        ],
        "Fireworks (Activity)": [
            3010, 4030, 4020, 3110, 4070
        ],
        "Flower Waterer (Activity)": [
            3010, 4070, 3110, 4030, 4020, 4025, 4022
        ],
        "Sandcastle Builder (Activity)": [
            3010, 3110, 4070, 4090, 4030,
            4035, 4021, 4020
        ],
        "Watering Hole (Activity)": [
            3010, 3110, 4021, 4020, 4025, 4070,
            5000, 5010
        ]
    }

    activity_columns_map: Dict[
        str, Dict[
            str, Union[
                str,
                List[
                    Union[str, bool]
                    ]
                ]
            ]
        ] = {
        "Bottle Filler (Activity)": {
            "round": "max",
            "identifier": [
                "addToYourCollection", "Dot_AllDoneTapThis",
                "ifYouWantToTry", "niceJob", "thatLooksSoCool",
                "oohStripes", "wowSoCool", "Dot_TrySomethingNew",
                "andItsFull", "ohWow", "dragABottle"],
            "jar_filled": [True, False]
        },
        "Bug Measurer (Activity)": {
            "identifier": [
                "sid_bugtank_line22",
                "sid_bugtank_line21", "sid_bugtank_line8_ALT",
                "sid_1", "sid_2", "sid_3", "sid_4", "sid_5",
                "sid_6", "sid_7", "sid_bugtank_line20",
                "sid_bugtank_line3", "Dot_TrySomethingNew"]
        },
        "Chicken Balancer (Activity)": {
            "identifier": [
                'morechicksheavier',
                'Dot_AllDoneTapThis',
                'dragchicks',
                'Dot_TrySomethingNew'],
            "layout.left.pig": [True, False],
            "layout.right.pig": [True, False]
        },
        "Egg Dropper (Activity)": {
            "identifier": [
                "Buddy_Incoming", "Buddy_MoreThanOneEgg",
                "Buddy_TapDino", "Buddy_TryDifferentNest",
                "Buddy_EggsWentToOtherNest"
            ]
        },
        "Fireworks (Activity)": {
            "identifier": [
                "Dot_GreatJob", "Dot_SoHigh",
                "Dot_Wow", "Dot_Amazing", "Dot_AllDoneTapThis",
                "Dot_SoLow", "Dot_WhoaSoCool", "Dot_GoLower",
                "Dot_TrySomethingNew", "Dot_UseFinger", "Dot_GoHigher"
            ],
            "launched": [True, False]
        },
        "Flower Waterer (Activity)": {
            "identifier": [
                "Dot_AllDoneTapThis", "basket",
                "tallflower", "plantastic", "leangreen",
                "greenthumb", "Dot_TrySomethingNew"
            ]
        },
        "Sandcastle Builder (Activity)": {
            "identifier": [
                "Dot_DragShovel", "Dot_AllDoneTapThis",
                "Dot_DragMoldPlace", "Dot_SoCool", "Dot_TryWall",
                "Dot_MoldSmall", "Dot_TrySomethingNew",
                "Dot_GreatJob", "Dot_FillItUp", "Dot_TryTower",
                "Dot_MoldBig"
            ],
            "filled": [True, False]
        },
        "Watering Hole (Activity)": {
            "identifier": [
                "Buddy_TaptoStartShower",
                "Dot_AllDoneTapThis", "Buddy_BigCloud",
                "Buddy_SmallCloud", "Buddy_KeepMakingItRain",
                "Dot_TrySomethingNew"
            ],
            "filled": [True, False]
        }
    }

    all_assessments: List[Dict[str, IoF]] = []
    past_activity_summarys: Dict[str, List[Dict[str, IoF]]] = {
        title: []
        for title in activity_codes_map.keys()
    }
    for sess_id, sess in user_sample.groupby("game_session", sort=False):
        if ((sess["type"].iloc[0] != "Assessment")
                and (sess["type"].iloc[0] != "Activity")):
            continue

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

        if sess["type"].iloc[0] == "Assessment" and (test or len(sess) > 1):
            features: Dict[str, IoF] = {}
            sess_title: str = sess.title.iloc[0]

            for key, summs in past_activity_summarys.items():
                if key == "Bottle Filler (Activity)":
                    if len(summs) == 0:
                        features["max_round"] = 0
                        features[key + "_duration"] = 0
                        features["mean_" + key + "_duration"] = 0
                        features["n_jar_filled_True"] = 0
                        features["n_jar_filled_False"] = 0
                        features["n_jar_filled_ratio"] = 0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = 0
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = 0
                    else:
                        features["max_round"] = max(
                            collect(summs, "max_round"))
                        features[key + "_duration"] = sum(
                            collect(summs, "duration"))
                        features["mean_" + key + "_duration"] = \
                            features[key + "_duration"] / features["max_round"]
                        features["n_jar_filled_True"] = sum(
                            collect(summs, "jar_filled_eq_True"))
                        features["n_jar_filled_False"] = sum(
                            collect(summs, "jar_filled_eq_False"))
                        total = features["n_jar_filled_False"] + \
                            features["n_jar_filled_True"]
                        features["n_jar_filled_ratio"] = \
                            features["n_jar_filled_True"] / total \
                            if total > 0 else 0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = \
                                sum(collect(
                                    summs, "identifier" + "_eq_" + str(ident)))
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = sum(
                                collect(summs, str(code))
                            )
                elif key == "Chicken Balancer (Activity)":
                    if len(summs) == 0:
                        features["n_layout.left.pig_True"] = 0
                        features["n_layout.right.pig_True"] = 0
                        features["n_layout.left.pig_False"] = 0
                        features["n_layout.right.pig_False"] = 0
                        features["layout.left.pig_ratio"] = 0.0
                        features["layout.right.pig_ratio"] = 0.0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_eq_" + str(ident) + "_count"] = 0
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = 0
                    else:
                        features["n_layout.left.pig_True"] = sum(
                            collect(summs, "layout.left.pig_eq_True"))
                        features["n_layout.left.pig_False"] = sum(
                            collect(summs, "layout.left.pig_eq_False"))
                        features["n_layout.right.pig_True"] = sum(
                            collect(summs, "layout.right.pig_eq_True"))
                        features["n_layout.right.pig_False"] = sum(
                            collect(summs, "layout.right.pig_eq_False"))
                        total = features["n_layout.left.pig_False"] + \
                            features["n_layout.left.pig_True"]
                        features["layout.left.pig_ratio"] = \
                            features["n_layout.left.pig_True"] / total \
                            if total > 0 else 0
                        total = features["n_layout.right.pig_False"] + \
                            features["n_layout.right.pig_True"]
                        features["layout.right.pig_ratio"] = \
                            features["n_layout.right.pig_True"] / total \
                            if total > 0 else 0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = \
                                sum(collect(
                                    summs, "identifier" + "_eq_" + str(ident)))
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = sum(
                                collect(summs, str(code))
                            )
                elif key == "Fireworks (Activity)":
                    if len(summs) == 0:
                        features["n_launched_True"] = 0
                        features["n_launched_False"] = 0
                        features["launched_ratio"] = 0.0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = 0
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = 0
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
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = \
                                sum(collect(
                                    summs, "identifier" + "_eq_" + str(ident)))
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = sum(
                                collect(summs, str(code))
                            )

                elif key == "Sandcastle Builder (Activity)":
                    if len(summs) == 0:
                        features["n_sand_filled_True"] = 0
                        features["n_sand_filled_False"] = 0
                        features["sand_filled_ratio"] = 0.0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = 0
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = 0
                    else:
                        features["n_sand_filled_True"] = sum(
                            collect(summs, "filled_eq_True"))
                        features["n_sand_filled_False"] = sum(
                            collect(summs, "filled_eq_False"))
                        total = features["n_sand_filled_False"] + \
                            features["n_sand_filled_True"]
                        features["sand_filled_ratio"] = \
                            features["n_sand_filled_True"] / total \
                            if total > 0 else 0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = \
                                sum(collect(
                                    summs, "identifier" + "_eq_" + str(ident)))
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = sum(
                                collect(summs, str(code))
                            )
                elif key == "Watering Hole (Activity)":
                    if len(summs) == 0:
                        features["n_water_filled_True"] = 0
                        features["n_water_filled_False"] = 0
                        features["water_filled_ratio"] = 0.0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = 0
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = 0
                    else:
                        features["n_water_filled_True"] = sum(
                            collect(summs, "filled_eq_True"))
                        features["n_water_filled_False"] = sum(
                            collect(summs, "filled_eq_False"))
                        total = features["n_water_filled_False"] + \
                            features["n_water_filled_True"]
                        features["water_filled_ratio"] = \
                            features["n_water_filled_True"] / total \
                            if total > 0 else 0
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = \
                                sum(collect(
                                    summs, "identifier" + "_eq_" + str(ident)))
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = sum(
                                collect(summs, str(code))
                            )
                else:
                    if len(summs) == 0:
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = 0
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = 0
                    else:
                        for ident in activity_columns_map[key]["identifier"]:
                            features[key + "_" + str(ident) + "_count"] = \
                                sum(collect(
                                    summs, "identifier" + "_eq_" + str(ident)))
                        for code in activity_codes_map[key]:
                            features[key + "_" + str(code)] = sum(
                                collect(summs, str(code))
                            )

            attempt_code = 4110 if (
                sess_title == "Bird Measurer (Assessment)"
            ) else 4100
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


def get_event_data(df: pd.DataFrame) -> pd.DataFrame:
    return pd.io.json.json_normalize(df.event_data.apply(json.loads))


def collect(lod: List[Dict[str, IoF]], name: str) -> List[IoF]:
    return [d[name] for d in lod]
