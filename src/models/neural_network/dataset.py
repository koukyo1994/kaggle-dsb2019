import numpy as np
import pandas as pd
import torch.utils.data as torchdata

from typing import List, Optional, Union

AoS = Union[np.ndarray, pd.Series]


class DSBDataset(torchdata.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 categorical_features: List[str],
                 y: Optional[AoS] = None):
        non_categorical = [
            col for col in df.columns if col not in categorical_features
        ]
        self.non_categorical = df[non_categorical].values
        self.categorical = df[categorical_features].values
        self.y = y

    def __len__(self):
        return len(self.categorical)

    def __getitem__(self, idx):
        categorical = self.categorical[idx, :]
        non_categorical = self.non_categorical[idx, :]
        if self.y is not None:
            y = self.y[idx]
            return non_categorical, categorical, y
        else:
            return non_categorical, categorical
