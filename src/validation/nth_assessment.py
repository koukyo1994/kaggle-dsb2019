import numpy as np
import pandas as pd

from typing import List


def get_assessment_number(valid_df: pd.DataFrame, test_df: pd.DataFrame):
    valid_nth_assessment: List[int] = []
    for _, sample in valid_df.groupby("installation_id"):
        valid_nth_assessment.extend(np.arange(len(sample)) + 1)

    valid_df_ = valid_df.copy()
    valid_df_["nth_assessment"] = valid_nth_assessment

    test_nth_assessment = []
    for inst_id in test_df["installation_id"].values:
        if inst_id in valid_df_["installation_id"].values:
            test_nth_assessment.append(
                valid_df_.query(f"installation_id == '{inst_id}'")
                ["nth_assessment"].max() + 1)
        else:
            test_nth_assessment.append(1)
    return np.array(test_nth_assessment)
