import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple


class DSBBase(nn.Module):
    def __init__(self, cat_dims: List[Tuple[int, int]], n_non_categorical: int,
                 emb_drop: float, drop: float):
        super().__init__()
        self.n_non_categorical = n_non_categorical

        self.embeddings = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in cat_dims])
        n_emb_out = sum([y for x, y in cat_dims])
        self.emb_drop = nn.Dropout(emb_drop)
        self.cat_bn = nn.BatchNorm1d(n_non_categorical + n_emb_out)
        self.lin1 = nn.Linear(n_non_categorical + n_emb_out, 150)
        self.bn1 = nn.BatchNorm1d(150)
        self.drop = nn.Dropout(drop)
        self.lin2 = nn.Linear(150, 50)
        self.bn2 = nn.BatchNorm1d(50)

    def forward(self, non_cat, cat) -> torch.Tensor:
        emb = [
            emb_layer(cat[:, j]) for j, emb_layer in enumerate(self.embeddings)
        ]
        emb = self.emb_drop(torch.cat(emb, 1))
        concat = torch.cat([non_cat, emb], 1)
        x = F.relu(self.bn1(self.lin1(concat)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.lin2(x)))
        return x


class DSBRegressor(nn.Module):
    def __init__(self, cat_dims: List[Tuple[int, int]], n_non_categorical: int,
                 **params):
        super().__init__()
        self.base = DSBBase(cat_dims, n_non_categorical, **params)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(50, 1)

    def forward(self, non_cat, cat) -> torch.Tensor:
        x = self.base(non_cat, cat)
        x = self.drop(x)
        x = F.relu(self.head(x))
        return torch.clamp(x.view(-1), 0.0, 1.0)


class DSBClassifier(nn.Module):
    def __init__(self, cat_dims: List[Tuple[int, int]], n_non_categorical: int,
                 **params):
        super().__init__()
        self.base = DSBBase(cat_dims, n_non_categorical, **params)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(50, 4)

    def forward(self, non_cat, cat) -> torch.Tensor:
        x = self.base(non_cat, cat)
        x = self.drop(x)
        x = F.softmax(self.head(x))
        return x


class DSBBinary(nn.Module):
    def __init__(self, cat_dims: List[Tuple[int, int]], n_non_categorical: int,
                 **params):
        super().__init__()
        self.base = DSBBase(cat_dims, n_non_categorical, **params)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(50, 1)

    def forward(self, non_cat, cat) -> torch.Tensor:
        x = self.base(non_cat, cat)
        x = self.drop(x)
        x = F.sigmoid(self.head(x))
        return x.view(-1)


class DSBOvR(nn.Module):
    def __init__(self, cat_dims: List[Tuple[int, int]], n_non_categorical: int,
                 **params):
        super().__init__()
        self.base = DSBBase(cat_dims, n_non_categorical, **params)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(50, 4)

    def forward(self, non_cat, cat) -> torch.Tensor:
        x = self.base(non_cat, cat)
        x = self.drop(x)
        x = F.sigmoid(self.head(x))
        return x


class DSBRegressionOvR(nn.Module):
    def __init__(self, cat_dims: List[Tuple[int, int]], n_non_categorical: int,
                 **params):
        super().__init__()
        self.base = DSBBase(cat_dims, n_non_categorical, **params)
        self.drop = nn.Dropout(0.3)
        self.regression_head = nn.Linear(50, 1)
        self.ovr_head = nn.Linear(50, 4)

    def forward(self, non_cat, cat):
        x = self.base(non_cat, cat)
        x = self.drop(x)
        x_regr = F.relu(self.regression_head(x)).view(-1)
        x_ovr = F.sigmoid(self.ovr_head(x))
        return torch.clamp(x_regr, 0.0, 1.0), x_ovr


class DSBRegressionBinary(nn.Module):
    def __init__(self, cat_dims: List[Tuple[int, int]], n_non_categorical: int,
                 **params):
        super().__init__()
        self.base = DSBBase(cat_dims, n_non_categorical, **params)
        self.drop = nn.Dropout(0.3)
        self.regression_head = nn.Linear(50, 1)
        self.binary_head = nn.Linear(50, 1)

    def forward(self, non_cat, cat):
        x = self.base(non_cat, cat)
        x = self.drop(x)
        x_regr = F.relu(self.regression_head(x)).view(-1)
        x_bin = F.sigmoid(self.binary_head(x))
        return torch.clamp(x_regr, 0.0, 1.0), x_bin.view(-1)
