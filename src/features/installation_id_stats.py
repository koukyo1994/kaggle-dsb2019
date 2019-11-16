import pandas as pd

from typing import Optional
from pathlib import Path

from src.features.base import Feature


class UnifiedWithInstallationIDStats(Feature):
    def create_features(self, train: Optional[pd.DataFrame],
                        test: Optional[pd.DataFrame]):
        # NOTE: this feature does not use train/test DataFrame

        self.train = pd.read_feather(
            Path(self.save_dir) / "ModifiedUnified_train.ftr")
        self.valid = pd.read_feather(
            Path(self.save_dir) / "ModifiedUnified_valid.ftr")
        self.test = pd.read_feather(
            Path(self.save_dir) / "ModifiedUnified_test.ftr")

        n_valid = len(self.valid)

        data_from_test = pd.concat(
            [self.valid.copy(), self.test.copy()],
            axis=0,
            sort=False).reset_index(drop=True)

        for df in [self.train, data_from_test]:
            df["installation_clip_count"] = df.groupby(
                "installation_id")["Clip"].transform("count")
            df["installation_game_count"] = df.groupby(
                "installation_id")["Game"].transform("count")
            df["installation_activity_count"] = df.groupby(
                "installation_id")["Activity"].transform("count")
            df["installation_duration_mean"] = df.groupby(
                "installation_id")["duration_mean"].transform("mean")
            df["installation_title_nunique"] = df.groupby(
                "installation_id")["session_title"].transform("nunique")
            df["sum_event_code_count"] = df[[
                str(i) for i in [
                    2050, 4100, 4230, 5000, 4235, 2060,
                    4110, 5010, 2070, 2075, 2080, 2081,
                    2083, 3110, 4010, 3120, 3121, 4020,
                    4021, 4022, 4025, 4030, 4031, 3010,
                    4035, 4040, 3020, 3021, 4045, 2000,
                    4050, 2010, 2020, 4070, 2025, 2030,
                    4080, 2035, 2040, 4090, 4220, 4095
                ]
            ]].sum(axis=1)
            df["installation_event_code_mean"] = df.groupby(
                "installation_id")["sum_event_code_count"].transform("mean")

        self.valid = data_from_test.loc[:n_valid-1, :].reset_index(drop=True)
        self.test = data_from_test.loc[n_valid:, :].reset_index(drop=True)
