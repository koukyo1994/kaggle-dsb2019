import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return ((y_pred - y_true)**2).mean().sqrt()


class RMSEBCELoss(nn.Module):
    def __init__(self, weights=(1.0, 1.0)):
        super().__init__()
        self.weights = weights
        self.bce = nn.BCELoss()

    def forward(self, y_pred_regr, y_true_regr, y_pred_ovr, y_true_ovr):
        rmse = ((y_pred_regr - y_true_regr)**2).mean().sqrt()
        bce = self.bce(y_pred_ovr, y_true_ovr)
        return self.weights[0] * rmse + self.weights[1] * bce


class RMSE2BCELoss(nn.Module):
    def __init__(self, weights=(1.0, 1.0)):
        super().__init__()
        self.weights = weights
        self.bce = nn.BCELoss()

    def forward(self, y_pred_regr, y_true_regr, y_pred_ovr, y_true_ovr,
                y_pred_bin, y_true_bin):
        rmse = ((y_pred_regr - y_true_regr)**2).mean().sqrt()
        bce1 = self.bce(y_pred_ovr, y_true_ovr)
        bce2 = self.bce(y_pred_bin, y_true_bin)
        return self.weights[0] * rmse + self.weights[1] * bce1 + self.weights[
            2] * bce2
