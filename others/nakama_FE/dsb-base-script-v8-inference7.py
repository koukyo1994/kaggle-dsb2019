# dataset taken from https://www.kaggle.com/yasufuminakama/dsb-base-script-v8-ratio/output
# ===========================================================
# Library
# ===========================================================
from fastai.tabular import *
from fastai.callbacks import *

import os
import gc
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from contextlib import contextmanager
from numba import jit
import time

import itertools
from functools import partial
from collections import Counter, defaultdict
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import pandas.tseries.offsets as offsets
import scipy as sp
from scipy import stats
from scipy.stats import kurtosis
from scipy.stats import skew
from math import sqrt
import random

import matplotlib.pyplot as plt
import seaborn as sns

import category_encoders as ce

from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn import preprocessing
from sklearn import metrics

import torch

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import warnings

warnings.filterwarnings("ignore")


# ===========================================================
# Utils
# ===========================================================
def get_logger(filename='log'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    # handler1
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    # handler2
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    # addHandler
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


logger = get_logger()


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.info(f'[{name}] done in {time.time() - t0:.0f} s')


def seed_everything(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_df(path, df_name, debug=False):
    # load df .csv or .pkl
    if path.split('.')[-1] == 'csv':
        df = pd.read_csv(path)
        if debug:
            df = pd.read_csv(path, nrows=1000)
    elif path.split('.')[-1] == 'pkl':
        df = pd.read_pickle(path)
    # output df shape
    if logger == None:
        print(f"{df_name} shape / {df.shape} ")
    else:
        logger.info(f"{df_name} shape / {df.shape} ")
    return df


def make_folds(_df, _id, target, fold, group=None, save_path='folds.csv'):
    df = _df.copy()
    if group == None:
        for n, (train_index, val_index) in enumerate(fold.split(df, df[target])):
            df.loc[val_index, 'fold'] = int(n)
    else:
        le = preprocessing.LabelEncoder()
        groups = le.fit_transform(df[group].values)
        for n, (train_index, val_index) in enumerate(fold.split(df, df[target], groups)):
            df.loc[val_index, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)
    df[[_id, 'game_session', target, 'fold']].to_csv(save_path, index=None)
    return df[[_id, 'game_session', target, 'fold']]


def make_stratified_group_k_folds(_df, _id, target, group, k, seed=42, save_path='folds.csv'):
    def stratified_group_k_fold(X, y, groups, k, seed=42):

        """
        original author : jakubwasikowski
        reference : https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
        """

        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)

        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(seed).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(k):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(k):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(groups) if g in test_groups]

            yield train_indices, test_indices

    df = _df.copy()
    le = preprocessing.LabelEncoder()
    groups = le.fit_transform(df[group].values)
    for n, (train_index, val_index) in enumerate(stratified_group_k_fold(df, df[target], groups, k=k, seed=seed)):
        df.loc[val_index, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)
    df[[_id, 'game_session', target, 'fold']].to_csv(save_path, index=None)

    return df[[_id, 'game_session', target, 'fold']]


@jit
def quadratic_weighted_kappa(a1, a2, max_rat=3):
    """
    https://www.kaggle.com/cpmpml/ultra-fast-qwk-calc-method
    """
    assert (len(a1) == len(a2))
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1,))
    hist2 = np.zeros((max_rat + 1,))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


def to_bins(x, borders):
    for i in range(len(borders)):
        if x <= borders[i]:
            return i
    return len(borders)


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _loss(self, coef, X, y, idx):
        X_p = np.array([to_bins(pred, coef) for pred in X])
        ll = -quadratic_weighted_kappa(y, X_p)
        return ll

    def fit(self, X, y):
        coef = [0.5, 1.5, 2.5]
        golden1 = 0.618
        golden2 = 1 - golden1
        ab_start = [(0.5, 1.3), (1.3, 2.0), (2.0, 2.5)]
        for it1 in range(10):
            for idx in range(3):
                # golden section search
                a, b = ab_start[idx]
                # calc losses
                coef[idx] = a
                la = self._loss(coef, X, y, idx)
                coef[idx] = b
                lb = self._loss(coef, X, y, idx)
                for it in range(20):
                    # choose value
                    if la > lb:
                        a = b - (b - a) * golden1
                        coef[idx] = a
                        la = self._loss(coef, X, y, idx)
                    else:
                        b = b - (b - a) * golden2
                        coef[idx] = b
                        lb = self._loss(coef, X, y, idx)
        self.coef_ = {'x': coef}

    def predict(self, X, coef):
        X_p = np.array([to_bins(pred, coef) for pred in X])
        return X_p

    def coefficients(self):
        return self.coef_['x']


def eval_with_truncated_data(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        groups: np.ndarray,
        n_trials: int = 10) -> Dict[str, Union[List[float], float]]:
    eval_result: Dict[str, Union[List[float], float]] = {}
    trials: List[float] = []

    index = np.arange(len(y_pred))
    gp_idx_df = pd.DataFrame({"groups": groups, "index": index})
    dice_results = []
    for _, df in gp_idx_df.groupby("groups"):
        dice_result = np.random.choice(df["index"], size=n_trials)
        dice_results.append(dice_result)

    idx_choice = np.vstack(dice_results)
    for i in range(n_trials):
        y_pred_choice = y_pred[idx_choice[:, i]]
        y_true_choice = y_true[idx_choice[:, i]]
        trials.append(quadratic_weighted_kappa(y_true_choice, y_pred_choice))

    mean_score = np.mean(trials)
    std = np.std(trials)
    eval_result["mean"] = mean_score
    eval_result["all_trials"] = trials
    lower_bound = mean_score - 2 * std
    upper_bound = mean_score + 2 * std

    logger.info(f'n_trials: {n_trials}')
    logger.info(f'mean_score: {mean_score}')
    logger.info(f'std: {std}')
    logger.info(f'0.95lower_bound: {lower_bound}')
    logger.info(f'0.95upper_bound: {upper_bound}')


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        logger.info('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                    start_mem - end_mem) / start_mem))
    return df


# ===========================================================
# Config
# ===========================================================
PARENT_DICT = '../input/data-science-bowl-2019/'
FEATURE_DICT = '../input/dsb-v8-ratio/'
df_path_dict = {'train': FEATURE_DICT + 'train_and_label_test.pkl',
                'test': PARENT_DICT + 'test.csv',
                'train_labels': PARENT_DICT + 'train_labels.csv',
                'specs': PARENT_DICT + 'specs.csv',
                'sample_submission': PARENT_DICT + 'sample_submission.csv'}
OUTPUT_DICT = ''

ID = 'installation_id'
TARGET = 'accuracy_group'
SEED = 42
seed_everything(seed=SEED)

N_FOLD = 5
FOLD_TYPE = 'StratifiedGroupKFold'
GROUP = 'installation_id'

DAYS = True


# ===========================================================
# Model
# ===========================================================
def run_single_lightgbm(param, train_df, test_df, folds, features, target, fold_num=0, categorical=[]):
    trn_idx = folds[folds.fold != fold_num].index
    val_idx = folds[folds.fold == fold_num].index
    logger.info(f'len(trn_idx) : {len(trn_idx)}')
    logger.info(f'len(val_idx) : {len(val_idx)}')

    if categorical == []:
        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features],
                               label=target.iloc[trn_idx])
        val_data = lgb.Dataset(train_df.iloc[val_idx][features],
                               label=target.iloc[val_idx])
    else:
        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features],
                               label=target.iloc[trn_idx],
                               categorical_feature=categorical)
        val_data = lgb.Dataset(train_df.iloc[val_idx][features],
                               label=target.iloc[val_idx],
                               categorical_feature=categorical)

    oof = np.zeros(len(train_df))
    predictions = np.zeros(len(test_df))

    num_round = 10000

    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=1000,
                    early_stopping_rounds=100)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_num

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration)

    # RMSE
    logger.info("fold{} RMSE score: {:<8.5f}"
                .format(fold_num, np.sqrt(metrics.mean_squared_error(target[val_idx], oof[val_idx]))))

    return oof, predictions, fold_importance_df


def run_kfold_lightgbm(param, train, test, folds, features, target, n_fold=5, categorical=[], version=None):
    logger.info(f"================================= {n_fold}fold lightgbm =================================")

    val_indexes = folds[folds.fold >= 0].index

    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    for fold_ in range(n_fold):
        print("Fold {}".format(fold_))
        _oof, _predictions, fold_importance_df = run_single_lightgbm(param,
                                                                     train,
                                                                     test,
                                                                     folds,
                                                                     features,
                                                                     target,
                                                                     fold_num=fold_,
                                                                     categorical=categorical)
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        oof += _oof
        predictions += _predictions / n_fold

    # RMSE
    logger.info("CV RMSE score: {:<8.5f}"
                .format(np.sqrt(metrics.mean_squared_error(target[val_indexes], oof[val_indexes]))))

    # QWK
    optR = OptimizedRounder()
    optR.fit(oof[val_indexes], target[val_indexes])
    coefficients = optR.coefficients()
    # coefficients = [0.5, 1.5, 2.5]
    logger.info(f"coefficients: {coefficients}")
    qwk_oof = optR.predict(oof[val_indexes], coefficients)
    logger.info("CV QWK score: {:<8.5f}"
                .format(quadratic_weighted_kappa(qwk_oof, target[val_indexes])))
    qwk_predictions = optR.predict(predictions, coefficients)

    if version==None:
        pd.DataFrame({'game_session': train.loc[val_indexes]['game_session'].values, f"{TARGET}": oof[val_indexes]}).to_csv(
            OUTPUT_DICT + 'oof_lightgbm_soft.csv', index=False)
        pd.DataFrame({'game_session': train.loc[val_indexes]['game_session'].values, f"{TARGET}": qwk_oof}).to_csv(
            OUTPUT_DICT + 'oof_lightgbm_hard.csv', index=False)
        pd.DataFrame({f"{ID}": test[ID].values, f"{TARGET}": predictions}) \
            .to_csv(OUTPUT_DICT + 'predictions_lightgbm_soft.csv', index=False)
        submission = pd.DataFrame({f"{ID}": test[ID].values, f"{TARGET}": qwk_predictions})
        submission[TARGET] = submission[TARGET].astype(int)
        submission.to_csv(OUTPUT_DICT + 'predictions_lightgbm_hard.csv', index=False)
        feature_importance_df.to_csv(OUTPUT_DICT + 'feature_importance_df_lightgbm.csv', index=False)
    else:
        pd.DataFrame({'game_session': train.loc[val_indexes]['game_session'].values, f"{TARGET}": oof[val_indexes]}).to_csv(
            OUTPUT_DICT + f'oof_lightgbm_soft{version}.csv', index=False)
        pd.DataFrame({'game_session': train.loc[val_indexes]['game_session'].values, f"{TARGET}": qwk_oof}).to_csv(
            OUTPUT_DICT + f'oof_lightgbm_hard{version}.csv', index=False)
        pd.DataFrame({f"{ID}": test[ID].values, f"{TARGET}": predictions}) \
            .to_csv(OUTPUT_DICT + f'predictions_lightgbm_soft{version}.csv', index=False)
        submission = pd.DataFrame({f"{ID}": test[ID].values, f"{TARGET}": qwk_predictions})
        submission[TARGET] = submission[TARGET].astype(int)
        submission.to_csv(OUTPUT_DICT + f'predictions_lightgbm_hard{version}.csv', index=False)
        feature_importance_df.to_csv(OUTPUT_DICT + f'feature_importance_df_lightgbm{version}.csv', index=False)

    logger.info(f"=========================================================================================")

    return feature_importance_df


def show_feature_importance(feature_importance_df, version=None):
    cols = (feature_importance_df[["Feature", "importance"]].groupby("Feature").mean()\
            .sort_values(by="importance", ascending=False)[:50].index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]
    plt.figure(figsize=(8, 16))
    sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('Features importance (averaged/folds)')
    plt.tight_layout()
    if version==None:
        plt.savefig(OUTPUT_DICT + 'feature_importance_df_lightgbm.png')
    else:
        plt.savefig(OUTPUT_DICT + f'feature_importance_df_lightgbm{version}.png')


def run_single_xgb(param, train_df, test_df, folds, features, target, fold_num=0):
    trn_idx = folds[folds.fold != fold_num].index
    val_idx = folds[folds.fold == fold_num].index

    d_train = xgb.DMatrix(data=train_df.iloc[trn_idx][features],
                          label=target.iloc[trn_idx])
    d_valid = xgb.DMatrix(data=train_df.iloc[val_idx][features],
                          label=target.iloc[val_idx])

    oof = np.zeros(len(train_df))
    predictions = np.zeros(len(test_df))

    num_round = 10000
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    clf = xgb.train(dtrain=d_train,
                    num_boost_round=num_round,
                    evals=watchlist,
                    early_stopping_rounds=100,
                    verbose_eval=1000,
                    params=param)

    oof[val_idx] = clf.predict(xgb.DMatrix(train_df.iloc[val_idx][features]),
                               ntree_limit=clf.best_ntree_limit)

    predictions += clf.predict(xgb.DMatrix(test_df[features]),
                               ntree_limit=clf.best_ntree_limit)

    logger.info("fold{} score: {:<8.5f}"
                .format(fold_num, np.sqrt(metrics.mean_squared_error(target[val_idx], oof[val_idx]))))

    return oof, predictions


def run_kfold_xgb(param, train, test, folds, features, target, n_fold=5, version=None):
    logger.info(f"================================= {n_fold}fold xgboost =================================")

    val_indexes = folds[folds.fold >= 0].index

    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))

    for fold_ in range(n_fold):
        print("Fold {}".format(fold_))
        _oof, _predictions = run_single_xgb(param,
                                            train,
                                            test,
                                            folds,
                                            features,
                                            target,
                                            fold_num=fold_)
        oof += _oof
        predictions += _predictions / n_fold

    logger.info("CV score: {:<8.5f}"
                .format(np.sqrt(metrics.mean_squared_error(target[val_indexes], oof[val_indexes]))))

    # QWK
    optR = OptimizedRounder()
    optR.fit(oof[val_indexes], target[val_indexes])
    coefficients = optR.coefficients()
    # coefficients = [0.5, 1.5, 2.5]
    logger.info(f"coefficients: {coefficients}")
    qwk_oof = optR.predict(oof[val_indexes], coefficients)
    logger.info("CV QWK score: {:<8.5f}"
                .format(quadratic_weighted_kappa(qwk_oof, target[val_indexes])))
    qwk_predictions = optR.predict(predictions, coefficients)

    if version==None:
        pd.DataFrame({'game_session': train.loc[val_indexes]['game_session'].values, f"{TARGET}": oof[val_indexes]}).to_csv(
            OUTPUT_DICT + 'oof_xgboost_soft.csv', index=False)
        pd.DataFrame({'game_session': train.loc[val_indexes]['game_session'].values, f"{TARGET}": qwk_oof}).to_csv(
            OUTPUT_DICT + 'oof_xgboost_hard.csv', index=False)
        pd.DataFrame({f"{ID}": test[ID].values, f"{TARGET}": predictions}).to_csv(
            OUTPUT_DICT + 'predictions_xgboost_soft.csv', index=False)
        submission = pd.DataFrame({f"{ID}": test[ID].values, f"{TARGET}": qwk_predictions})
        submission[TARGET] = submission[TARGET].astype(int)
        submission.to_csv(OUTPUT_DICT + 'predictions_xgboost_hard.csv', index=False)
    else:
        pd.DataFrame({'game_session': train.loc[val_indexes]['game_session'].values, f"{TARGET}": oof[val_indexes]}).to_csv(
            OUTPUT_DICT + f'oof_xgboost_soft{version}.csv', index=False)
        pd.DataFrame({'game_session': train.loc[val_indexes]['game_session'].values, f"{TARGET}": qwk_oof}).to_csv(
            OUTPUT_DICT + f'oof_xgboost_hard{version}.csv', index=False)
        pd.DataFrame({f"{ID}": test[ID].values, f"{TARGET}": predictions}).to_csv(
            OUTPUT_DICT + f'predictions_xgboost_soft{version}.csv', index=False)
        submission = pd.DataFrame({f"{ID}": test[ID].values, f"{TARGET}": qwk_predictions})
        submission[TARGET] = submission[TARGET].astype(int)
        submission.to_csv(OUTPUT_DICT + f'predictions_xgboost_hard{version}.csv', index=False)

    logger.info(f"========================================================================================")


def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]


def run_single_catboost(param, train_df, test_df, folds, features, target, fold_num=0, categorical=[]):
    trn_idx = folds[folds.fold != fold_num].index
    val_idx = folds[folds.fold == fold_num].index

    y_tr = target.iloc[trn_idx].values
    X_tr = train_df.iloc[trn_idx][features]
    y_val = target.iloc[val_idx].values
    X_val = train_df.iloc[val_idx][features]

    categorical_features_pos = column_index(X_tr, categorical)

    oof = np.zeros(len(train_df))
    predictions = np.zeros(len(test_df))

    clf = cb.CatBoostRegressor(**param)
    clf.fit(X_tr, y_tr, eval_set=(X_val, y_val),
            cat_features=categorical_features_pos,
            use_best_model=True)

    oof[val_idx] = clf.predict(X_val)
    predictions += clf.predict(test_df[features])

    logger.info("fold{} score: {:<8.5f}"
                .format(fold_num, np.sqrt(metrics.mean_squared_error(target[val_idx], oof[val_idx]))))

    return oof, predictions


def run_kfold_catboost(param, train, test, folds, features, target, n_fold=5, categorical=[], version=None):
    logger.info(f"================================= {n_fold}fold catboost =================================")

    val_indexes = folds[folds.fold >= 0].index

    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))

    for fold_ in range(n_fold):
        print("Fold {}".format(fold_))
        _oof, _predictions = run_single_catboost(param,
                                                 train,
                                                 test,
                                                 folds,
                                                 features,
                                                 target,
                                                 fold_num=fold_,
                                                 categorical=categorical)
        oof += _oof
        predictions += _predictions / n_fold

    logger.info("CV score: {:<8.5f}"
                .format(np.sqrt(metrics.mean_squared_error(target[val_indexes], oof[val_indexes]))))

    # QWK
    optR = OptimizedRounder()
    optR.fit(oof[val_indexes], target[val_indexes])
    coefficients = optR.coefficients()
    # coefficients = [0.5, 1.5, 2.5]
    logger.info(f"coefficients: {coefficients}")
    qwk_oof = optR.predict(oof[val_indexes], coefficients)
    logger.info("CV QWK score: {:<8.5f}"
                .format(quadratic_weighted_kappa(qwk_oof, target[val_indexes])))
    qwk_predictions = optR.predict(predictions, coefficients)

    if version==None:
        pd.DataFrame({'game_session': train.loc[val_indexes]['game_session'].values, f"{TARGET}": oof[val_indexes]}).to_csv(
            OUTPUT_DICT + 'oof_catboost_soft.csv', index=False)
        pd.DataFrame({'game_session': train.loc[val_indexes]['game_session'].values, f"{TARGET}": qwk_oof}).to_csv(
            OUTPUT_DICT + 'oof_catboost_hard.csv', index=False)
        pd.DataFrame({f"{ID}": test[ID].values, f"{TARGET}": predictions}).to_csv(
            OUTPUT_DICT + 'predictions_catboost_soft.csv', index=False)
        submission = pd.DataFrame({f"{ID}": test[ID].values, f"{TARGET}": qwk_predictions})
        submission[TARGET] = submission[TARGET].astype(int)
        submission.to_csv(OUTPUT_DICT + 'predictions_catboost_hard.csv', index=False)
    else:
        pd.DataFrame({'game_session': train.loc[val_indexes]['game_session'].values, f"{TARGET}": oof[val_indexes]}).to_csv(
            OUTPUT_DICT + f'oof_catboost_soft{version}.csv', index=False)
        pd.DataFrame({'game_session': train.loc[val_indexes]['game_session'].values, f"{TARGET}": qwk_oof}).to_csv(
            OUTPUT_DICT + f'oof_catboost_hard{version}.csv', index=False)
        pd.DataFrame({f"{ID}": test[ID].values, f"{TARGET}": predictions}).to_csv(
            OUTPUT_DICT + f'predictions_catboost_soft{version}.csv', index=False)
        submission = pd.DataFrame({f"{ID}": test[ID].values, f"{TARGET}": qwk_predictions})
        submission[TARGET] = submission[TARGET].astype(int)
        submission.to_csv(OUTPUT_DICT + f'predictions_catboost_hard{version}.csv', index=False)

    logger.info(f"=========================================================================================")


n_cat_imp = 1  # ['title']
n_cont_imp = 0
n_imp = n_cat_imp + n_cont_imp


class my_TabularModel(nn.Module):
    "Basic model for tabular data."

    def __init__(self, emb_szs: ListSizes, n_cont: int, out_sz: int, layers: Collection[int],
                 ps: Collection[float] = None,
                 emb_drop: float = 0., y_range: OptRange = None, use_bn: bool = True, bn_final: bool = False):
        super().__init__()
        ps = ifnone(ps, [0] * len(layers))
        ps = listify(ps, layers)
        self.embeds = nn.ModuleList([embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(emb_drop)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb, self.n_cont, self.y_range = n_emb, n_cont, y_range
        sizes = self.get_sizes(layers, out_sz)
        actns = [nn.ReLU(inplace=True) for _ in range(len(sizes) - 2)] + [None]
        if n_imp != 0:
            self.linear1 = nn.Linear(sizes[0] - n_imp, sizes[1])
            self.bn1 = nn.BatchNorm1d(sizes[1])
            self.linear2 = nn.Linear(sizes[1], sizes[2])
            self.bn2 = nn.BatchNorm1d(sizes[2])
            self.linear3 = nn.Linear(sizes[2] + n_imp, sizes[3])
        else:
            layers = []
            for i, (n_in, n_out, dp, act) in enumerate(zip(sizes[:-1], sizes[1:], [0.] + ps, actns)):
                layers += bn_drop_lin(n_in, n_out, bn=use_bn and i != 0, p=dp, actn=act)
            if bn_final:
                layers.append(nn.BatchNorm1d(sizes[-1]))
            self.layers = nn.Sequential(*layers)

    def get_sizes(self, layers, out_sz):
        return [self.n_emb + self.n_cont + n_imp] + layers + [out_sz]

    def forward(self, x_cat: Tensor, x_cont: Tensor) -> Tensor:
        if self.n_emb != 0:
            x = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            if n_cat_imp != 0:
                x_cat_imp = x[:, :n_cat_imp]
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        if n_imp != 0:
            x = F.relu(self.linear1(x))
            x = self.bn1(x)
            x = F.relu(self.linear2(x))
            x = self.bn2(x)
            if n_cat_imp != 0:
                x = torch.cat([x, x_cat_imp], 1)
            if n_cont_imp != 0:
                x = torch.cat([x, x_cont[:, :n_cont_imp]], 1)
            x = self.linear3(x)
        else:
            x = self.layers(x)
        if self.y_range is not None:
            x = (self.y_range[1] - self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
        return x


def run_single_nn(param, train, test, folds, cat_names, cont_names, procs, target, fold_num=0, seed=777):
    # Set seed
    seed_everything(seed=seed)

    # params
    batch_size = param['batch_size']
    epochs = param['epochs']
    lr = param['lr']
    emb_drop = param['emb_drop']

    # validation index
    val_idx = folds[folds.fold == fold_num].index

    # Test tabularlist
    test = TabularList.from_df(test, cat_names=cat_names, cont_names=cont_names, procs=procs)

    # Train data bunch
    data = (TabularList.from_df(train, path='.', cat_names=cat_names, cont_names=cont_names, procs=procs)
            .split_by_idx(val_idx)
            .label_from_df(cols=target, label_cls=FloatList)
            .add_test(test)
            .databunch(bs=batch_size))

    # Create deep learning model
    # BASIC
    # learn = tabular_learner(data, layers=[200,100], metrics=rmse, emb_drop=emb_drop)
    # MODIFIED
    emb_szs: Dict[str, int] = None
    emb_szs = data.get_emb_szs(ifnone(emb_szs, {}))
    model = my_TabularModel(emb_szs=emb_szs, n_cont=len(data.cont_names),
                            out_sz=data.c, layers=[200, 100], ps=[0., 0.], emb_drop=emb_drop)
    learn = Learner(data, model, metrics=rmse)
    if fold_num == 0:
        logger.info(learn.model)

    # Fit the model based on selected learning rate
    learn.fit(epochs, lr=lr,
              callbacks=[ReduceLROnPlateauCallback(learn, monitor='root_mean_squared_error', mode='min',
                                                   patience=3, factor=0.5, min_lr=1e-5),
                         SaveModelCallback(learn, every='improvement', monitor='root_mean_squared_error',
                                           mode='min', name=f"nn_fold{fold_num}")])

    # get oof predictions
    oof = np.zeros(len(train))
    _oof, _ = learn.get_preds()
    oof[val_idx] = np.array([p[0].data.item() for p in _oof])

    # get test predictions
    predictions, _ = learn.get_preds(DatasetType.Test)
    predictions = np.array([p[0].data.item() for p in predictions])

    logger.info("fold{} score: {:<8.5f}".format(fold_num,
                                                np.sqrt(metrics.mean_squared_error(train.loc[val_idx][target].values,
                                                                                   oof[val_idx]))))

    # del
    del test, data, learn
    gc.collect()

    return oof, predictions


def run_kfold_nn(nn_param, train, test, folds, cat_names, cont_names, procs, target, n_fold=5, seed=777, version=None):
    logger.info(f"================================= START {n_fold}fold nn =================================")

    val_indexes = folds[folds.fold >= 0].index

    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))

    for fold_ in range(n_fold):
        print("Fold {}".format(fold_))
        _oof, _predictions = run_single_nn(nn_param,
                                           train,
                                           test,
                                           folds,
                                           cat_names,
                                           cont_names,
                                           procs,
                                           TARGET,
                                           fold_num=fold_,
                                           seed=seed)
        oof += _oof
        predictions += _predictions / n_fold

    logger.info("CV score: {:<8.5f}"
                .format(np.sqrt(metrics.mean_squared_error(train.loc[val_indexes][target].values, oof[val_indexes]))))

    # QWK
    optR = OptimizedRounder()
    optR.fit(oof[val_indexes], train.loc[val_indexes][target].values)
    coefficients = optR.coefficients()
    # coefficients = [0.5, 1.5, 2.5]
    logger.info(f"coefficients: {coefficients}")
    qwk_oof = optR.predict(oof[val_indexes], coefficients)
    logger.info("CV QWK score: {:<8.5f}"
                .format(quadratic_weighted_kappa(qwk_oof, train.loc[val_indexes][target].values)))
    qwk_predictions = optR.predict(predictions, coefficients)
    
    if version==None:
        pd.DataFrame({'game_session': train.loc[val_indexes]['game_session'].values, f"{TARGET}": oof[val_indexes]}).to_csv(
            OUTPUT_DICT + 'oof_nn_soft.csv', index=False)
        pd.DataFrame({'game_session': train.loc[val_indexes]['game_session'].values, f"{TARGET}": qwk_oof}).to_csv(
            OUTPUT_DICT + 'oof_nn_hard.csv', index=False)
        pd.DataFrame({f"{ID}": test[ID].values, f"{TARGET}": predictions}).to_csv(OUTPUT_DICT + 'predictions_nn_soft.csv',
                                                                                  index=False)
        submission = pd.DataFrame({f"{ID}": test[ID].values, f"{TARGET}": qwk_predictions})
        submission[TARGET] = submission[TARGET].astype(int)
        submission.to_csv(OUTPUT_DICT + 'predictions_nn_hard.csv', index=False)
    else:
        pd.DataFrame({'game_session': train.loc[val_indexes]['game_session'].values, f"{TARGET}": oof[val_indexes]}).to_csv(
            OUTPUT_DICT + f'oof_nn_soft{version}.csv', index=False)
        pd.DataFrame({'game_session': train.loc[val_indexes]['game_session'].values, f"{TARGET}": qwk_oof}).to_csv(
            OUTPUT_DICT + f'oof_nn_hard{version}.csv', index=False)
        pd.DataFrame({f"{ID}": test[ID].values, f"{TARGET}": predictions}).to_csv(OUTPUT_DICT + f'predictions_nn_soft{version}.csv',
                                                                                  index=False)
        submission = pd.DataFrame({f"{ID}": test[ID].values, f"{TARGET}": qwk_predictions})
        submission[TARGET] = submission[TARGET].astype(int)
        submission.to_csv(OUTPUT_DICT + f'predictions_nn_hard{version}.csv', index=False)

    logger.info(f"=========================================================================================")


def load_oof_and_preds(oof_path_dict, predictions_path_dict):

    for i, path in enumerate(oof_path_dict):
        if i==0:
            oof_df = load_df(path=oof_path_dict[path], df_name=path)
            oof_df.columns = ['game_session', path]
        else:
            oof = load_df(path=oof_path_dict[path], df_name=path)
            oof.columns = ['game_session', path]
            oof_df = oof_df.merge(oof, on='game_session')

    for i, path in enumerate(predictions_path_dict):
        if i==0:
            predictions_df = load_df(path=predictions_path_dict[path], df_name=path)
            predictions_df.columns = [ID, path]
        else:
            predictions = load_df(path=predictions_path_dict[path], df_name=path)
            predictions.columns = [ID, path]
            predictions_df = predictions_df.merge(predictions, on=ID)

    features = [c for c in predictions_df.columns if c not in [ID]]

    return oof_df, predictions_df, features


def weight_averaging(predictions_df, features, best_weight, QWK=True, coefficients=[0.5, 1.5, 2.5], version=None):
    predictions = []
    for c in features:
        predictions.append(np.array(predictions_df.loc[:, c]))

    predictions_df[TARGET] = 0
    for i, c in enumerate(features):
        w = best_weight[i]
        predictions_df[TARGET] += np.array(predictions_df.loc[:, c]) * w

    if QWK:
        if version==None:
            predictions_df.to_csv('predictions_averaging_soft.csv')
        else:
            predictions_df.to_csv(f'predictions_averaging_soft{version}.csv')
        # QWK
        optR = OptimizedRounder()
        predictions_df[TARGET] = optR.predict(predictions_df[TARGET], coefficients)
        predictions_df[TARGET] = predictions_df[TARGET].astype(int)

    return predictions_df


def search_averaging_weight(oof_df, features, target, solver='L-BFGS-B', trials=100, version=None):
    predictions = []
    lls = []
    wghts = []

    def rmse_func(weights):
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
            final_prediction += weight * prediction
        return np.sqrt(metrics.mean_squared_error(target, final_prediction))

    for c in features:
        predictions.append(np.array(oof_df.loc[:, c]))

    for i in range(trials):
        seed_everything(seed=i)
        starting_values = np.random.uniform(size=len(features))
        cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
        bounds = [(0, 1)] * len(predictions)
        res = sp.optimize.minimize(rmse_func, starting_values, constraints=cons, bounds=bounds, method=solver)
        lls.append(res['fun'])
        wghts.append(res['x'])

    bestLoss = np.min(lls)
    bestWght = wghts[np.argmin(lls)]

    logger.info('CV Score: {best_loss:.7f}'.format(best_loss=bestLoss))
    logger.info('Best Weights: {weights:}'.format(weights=bestWght))

    oof = weight_averaging(oof_df, features, bestWght, QWK=False)

    # QWK
    optR = OptimizedRounder()
    optR.fit(oof[TARGET], target)
    coefficients = optR.coefficients()
    # coefficients = [0.5, 1.5, 2.5]
    logger.info(f"coefficients: {coefficients}")
    qwk_oof = optR.predict(oof[TARGET], coefficients)
    logger.info("CV QWK score: {:<8.5f}".format(quadratic_weighted_kappa(qwk_oof, target)))

    if version==None:
        pd.DataFrame({'game_session': oof_df['game_session'].values, f"{TARGET}": oof[TARGET]}).to_csv(
            OUTPUT_DICT + 'oof_averaging_soft.csv', index=False)
        pd.DataFrame({'game_session': oof_df['game_session'].values, f"{TARGET}": qwk_oof}).to_csv(
            OUTPUT_DICT + 'oof_averaging_hard.csv', index=False)
    else:
        pd.DataFrame({'game_session': oof_df['game_session'].values, f"{TARGET}": oof[TARGET]}).to_csv(
            OUTPUT_DICT + f'oof_averaging_soft{version}.csv', index=False)
        pd.DataFrame({'game_session': oof_df['game_session'].values, f"{TARGET}": qwk_oof}).to_csv(
            OUTPUT_DICT + f'oof_averaging_hard{version}.csv', index=False)

    return bestWght, coefficients


def model_pipeline(train, test, folds, features, cat_features, target, 
                   lgb_param, xgb_param, cat_param, nn_param, version=None, submission_file='submission.csv', DROP=True):
    if DROP:
        with timer('Drop useless columns'):
            feature_importance_df = run_kfold_lightgbm(lgb_param, train, test, folds, features, target,
                                                       n_fold=N_FOLD, categorical=cat_features, version=version)
            feature_importance_df = feature_importance_df[["Feature", "importance"]].groupby("Feature").mean()
            if version==None:
                feature_importance_df[feature_importance_df["importance"] == 0].reset_index()\
                    .to_csv('useless_features.csv', index=False)
            else:
                feature_importance_df[feature_importance_df["importance"] == 0].reset_index()\
                    .to_csv(f'useless_features{version}.csv', index=False)
            useless_features = feature_importance_df[feature_importance_df["importance"] == 0].index
            features = [c for c in features if c not in useless_features]
        
    with timer('Run LGBM'):
        feature_importance_df = run_kfold_lightgbm(lgb_param, train, test, folds, features, target,
                                                   n_fold=N_FOLD, categorical=cat_features, version=version)
        show_feature_importance(feature_importance_df, version=version)

    with timer('Run XGB'):
        run_kfold_xgb(xgb_param, train, test, folds, features, target, n_fold=N_FOLD, version=version)

    with timer('Run Catboost'):
        run_kfold_catboost(cat_param, train, test, folds, features, target, n_fold=N_FOLD, categorical=cat_features, version=version)

    with timer('Run NN'):
        nn_log_feats = [c for c in features if c.find('Counter')>=0]
        nn_log_feats = [c for c in nn_log_feats if not c.find('Ratio')>=0]
        for c in nn_log_feats:
            train[c] = np.log1p(train[c])
            test[c] = np.log1p(test[c])
        procs = [Categorify]
        cat_important_names = ['title']
        cat_names = ['world']
        cat_names = cat_important_names + cat_names
        cont_important_names = []
        cont_names = [c for c in features if c not in cat_names]
        cont_names = cont_important_names + cont_names
        run_kfold_nn(nn_param, train.fillna(-1), test.fillna(-1), folds, cat_names, cont_names, procs, 
                     TARGET, n_fold=N_FOLD, version=version)

    with timer('Run Averaging'):
        if version==None:
            oof_path_dict = {'lightgbm': OUTPUT_DICT + 'oof_lightgbm_soft.csv',
                             'xgboost': OUTPUT_DICT + 'oof_xgboost_soft.csv',
                             'catboost': OUTPUT_DICT + 'oof_catboost_soft.csv',
                             'nn': OUTPUT_DICT + 'oof_nn_soft.csv',
                             }
            predictions_path_dict = {'lightgbm': OUTPUT_DICT + 'predictions_lightgbm_soft.csv',
                                     'xgboost': OUTPUT_DICT + 'predictions_xgboost_soft.csv',
                                     'catboost': OUTPUT_DICT + 'predictions_catboost_soft.csv',
                                     'nn': OUTPUT_DICT + 'predictions_nn_soft.csv',
                                     }
        else:
            oof_path_dict = {'lightgbm': OUTPUT_DICT + f'oof_lightgbm_soft{version}.csv',
                             'xgboost': OUTPUT_DICT + f'oof_xgboost_soft{version}.csv',
                             'catboost': OUTPUT_DICT + f'oof_catboost_soft{version}.csv',
                             'nn': OUTPUT_DICT + f'oof_nn_soft{version}.csv',
                             }
            predictions_path_dict = {'lightgbm': OUTPUT_DICT + f'predictions_lightgbm_soft{version}.csv',
                                     'xgboost': OUTPUT_DICT + f'predictions_xgboost_soft{version}.csv',
                                     'catboost': OUTPUT_DICT + f'predictions_catboost_soft{version}.csv',
                                     'nn': OUTPUT_DICT + f'predictions_nn_soft{version}.csv',
                                     }
        oof_df, predictions_df, features = load_oof_and_preds(oof_path_dict, predictions_path_dict)
        _folds = folds[folds.fold >= 0].reset_index(drop=True)
        _folds = _folds.merge(oof_df, on='game_session')
        target = _folds[TARGET].copy()
        best_weight, coefficients = search_averaging_weight(_folds, features, target, 
                                                            solver='L-BFGS-B', trials=100, version=version)
        predictions_df = weight_averaging(predictions_df, features, best_weight, 
                                          QWK=True, coefficients=coefficients, version=version)
        predictions_df[[ID, TARGET]].to_csv(submission_file, index=False)


# ===========================================================
# Feature Engineering
# ===========================================================
def assessment(df):
    df['num_correct'] = 0
    df['num_incorrect'] = 0
    df.loc[((df.event_code == 4100) & (df.title != 'Bird Measurer (Assessment)')) & (
                df.type == 'Assessment'), 'num_correct'] = \
        df.loc[((df.event_code == 4100) & (df.title != 'Bird Measurer (Assessment)')) & (df.type == 'Assessment')] \
            ['event_data'].apply(lambda x: x.find('"correct":true') >= 0) * 1
    df.loc[((df.event_code == 4110) & (df.title == 'Bird Measurer (Assessment)')) & (
                df.type == 'Assessment'), 'num_correct'] = \
        df.loc[((df.event_code == 4110) & (df.title == 'Bird Measurer (Assessment)')) & (df.type == 'Assessment')] \
            ['event_data'].apply(lambda x: x.find('"correct":true') >= 0) * 1
    df.loc[((df.event_code == 4100) & (df.title != 'Bird Measurer (Assessment)')) & (
                df.type == 'Assessment'), 'num_incorrect'] = \
        df.loc[((df.event_code == 4100) & (df.title != 'Bird Measurer (Assessment)')) & (df.type == 'Assessment')] \
            ['event_data'].apply(lambda x: x.find('"correct":false') >= 0) * 1
    df.loc[((df.event_code == 4110) & (df.title == 'Bird Measurer (Assessment)')) & (
                df.type == 'Assessment'), 'num_incorrect'] = \
        df.loc[((df.event_code == 4110) & (df.title == 'Bird Measurer (Assessment)')) & (df.type == 'Assessment')] \
            ['event_data'].apply(lambda x: x.find('"correct":false') >= 0) * 1

    return df


def create_test_labels(test, sample_submission):
    # assessment
    cols = ['installation_id', 'game_session', 'title', 'num_correct', 'num_incorrect']
    test_labels = pd.concat([
        test[((test.event_code == 4100) & (test.title != 'Bird Measurer (Assessment)')) & (test.type == 'Assessment')] \
            [cols].groupby(['installation_id', 'game_session', 'title'], as_index=False).sum(),
        test[((test.event_code == 4110) & (test.title == 'Bird Measurer (Assessment)')) & (test.type == 'Assessment')] \
            [cols].groupby(['installation_id', 'game_session', 'title'], as_index=False).sum()
    ])
    test_labels['accuracy'] = test_labels['num_correct'] / (test_labels['num_correct'] + test_labels['num_incorrect'])
    test_labels['accuracy_group'] = np.nan
    test_labels.loc[(test_labels['num_correct'] == 1) & (test_labels['num_incorrect'] == 0), 'accuracy_group'] = 3
    test_labels.loc[(test_labels['num_correct'] == 1) & (test_labels['num_incorrect'] == 1), 'accuracy_group'] = 2
    test_labels.loc[(test_labels['num_correct'] == 1) & (test_labels['num_incorrect'] >= 2), 'accuracy_group'] = 1
    test_labels.loc[(test_labels['num_correct'] == 0), 'accuracy_group'] = 0
    test_labels['accuracy_group'] = test_labels['accuracy_group'].astype(int)

    # no assessment ( what we have to predict )
    key_cols = [ID, 'timestamp', 'event_code', 'type']
    last_assesment = test[test.event_code == 2000][test.type == 'Assessment'][key_cols].groupby(ID,
                                                                                                as_index=False).max()
    last_assesment_df = last_assesment.merge(test[key_cols + ['game_session', 'title']], on=key_cols, how='left') \
        [['installation_id', 'game_session', 'title']]

    # concat them
    test_labels = pd.concat([test_labels, last_assesment_df]).reset_index(drop=True)

    # drop ['num_correct', 'num_incorrect'] after assessment
    test = test.drop(columns=['num_correct', 'num_incorrect']).reset_index(drop=True)

    return test, test_labels


def extract_time_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['year'] = df['timestamp'].dt.year
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['weekofyear'] = df['timestamp'].dt.weekofyear

    df['timestamp'] = df['timestamp'].astype(int)

    df['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), df['title'], df['event_code']))

    return df


def extract_user_logs(df, df_labels):
    logs = pd.DataFrame()
    nunique_cols = ['event_id', 'game_session', 'timestamp', 'event_data', 'event_count', 'event_code',
                    'title', 'world', 'date', 'month', 'hour', 'dayofweek', 'weekofyear', 'title_event_code']
    sum_cols = ['title_event_code', 'title', 'event_code', 'world', 'type']
    sum_values = set()
    for c in sum_cols:
        sum_values = sum_values | set(df[c].unique())

    def extract_user_log(tmp, tmp_df, days=None):

        sum_df = pd.DataFrame()
        if days == None:
            _sum_df = Counter({value: 0 for value in list(sum_values)})
            for i in range(len(tmp_df)):
                if i == 0:
                    tmp_past = tmp[tmp.timestamp < tmp_df.loc[i, 'timestamp']]
                else:
                    tmp_past = tmp[tmp_df.loc[i - 1, 'timestamp'] <= tmp.timestamp][
                        tmp.timestamp < tmp_df.loc[i, 'timestamp']]
                if len(tmp_past) == 0:
                    sum_df = pd.concat([sum_df, pd.DataFrame({'No_playing_logs': [1]})], axis=0)
                else:
                    nunique_df = pd.DataFrame(tmp[tmp.timestamp < tmp_df.loc[i, 'timestamp']] \
                                                  [nunique_cols].nunique()).T.add_prefix('nunique_')
                    for c in sum_cols:
                        _sum_df.update(Counter(tmp_past[c].values))
                    concat_df = pd.concat([nunique_df,
                                           pd.DataFrame.from_dict(_sum_df, orient='index').T.add_suffix('_Counter')], axis=1)
                    sum_df = pd.concat([sum_df, concat_df], axis=0)
        else:
            past_days = days * 24 * 60 ** 2 * 10 ** 9
            for i in range(len(tmp_df)):
                if i == 0:
                    tmp_past = tmp[(tmp_df.loc[i, 'timestamp'] - past_days) < tmp.timestamp] \
                        [tmp.timestamp < tmp_df.loc[i, 'timestamp']]
                if len(tmp_past) == 0:
                    sum_df = pd.concat([sum_df, pd.DataFrame({f'{days}day_No_playing_logs': [1]})], axis=0)
                else:
                    nunique_df = pd.DataFrame(tmp_past[nunique_cols].nunique()).T.add_prefix(f'nunique_{days}day_')
                    _sum_df = Counter({value: 0 for value in list(sum_values)})
                    for c in sum_cols:
                        _sum_df.update(Counter(tmp_past[c].values))
                    concat_df = pd.concat([nunique_df,
                                           pd.DataFrame.from_dict(_sum_df, orient='index').T.add_suffix('_Counter')], axis=1).add_prefix(f'{days}day_')
                    sum_df = pd.concat([sum_df, concat_df], axis=0)

        return sum_df

    for (_, tmp) in df.groupby('installation_id'):
        tmp = tmp.sort_values('timestamp').reset_index(drop=True)
        tmp_df = tmp[tmp.event_code == 2000][tmp.type == 'Assessment'].reset_index(drop=True)
        sum_df = extract_user_log(tmp, tmp_df, days=None)

        # concat
        _log = pd.concat([tmp_df, sum_df.reset_index(drop=True)], axis=1)
        logs = pd.concat([logs, _log], axis=0)

    not_merge_columns = ['installation_id', 'title']
    output = df_labels.merge(logs.drop(columns=not_merge_columns), on='game_session', how='left')

    return output.reset_index(drop=True)


def past_solved_features(df):
    output = pd.DataFrame()
    target_cols = ['num_correct', 'num_incorrect', 'accuracy_group']
    title_cols = ['Cart Balancer (Assessment)', 'Cauldron Filler (Assessment)',
                  'Mushroom Sorter (Assessment)', 'Chest Sorter (Assessment)', 'Bird Measurer (Assessment)']

    def past_solved_feature(tmp, days=None):
        for i in range(len(tmp)):
            if i != 0:
                if days == None:
                    tmp_past = tmp[tmp.timestamp < tmp.loc[i, 'timestamp']]
                    if len(tmp_past) != 0:
                        for c in target_cols:
                            tmp_past_values = tmp_past[c].values
                            tmp.loc[i, c + '_sum'] = tmp_past_values.sum()
                            tmp.loc[i, c + '_max'] = tmp_past_values.max()
                            tmp.loc[i, c + '_min'] = tmp_past_values.min()
                            tmp.loc[i, c + '_mean'] = tmp_past_values.mean()
                            tmp.loc[i, c + '_median'] = tmp_past[c].median()
                            tmp.loc[i, c + '_var'] = tmp_past_values.var()
                            tmp.loc[i, c + '_last'] = tmp_past_values[-1]
                        tmp.loc[i, 'total_accuracy'] = \
                            tmp.loc[i, 'num_correct_sum'] / (
                                        tmp.loc[i, 'num_correct_sum'] + tmp.loc[i, 'num_incorrect_sum'])
                    # Bird Measurer, Cart Balancer, Cauldron Filler, Chest Sorter, and Mushroom Sorter
                    for t in title_cols:
                        _tmp_past = tmp_past[tmp_past.title == t]
                        if len(_tmp_past) != 0:
                            for c in target_cols:
                                tmp_past_values = _tmp_past[c].values
                                tmp.loc[i, c + '_sum_' + t] = tmp_past_values.sum()
                                tmp.loc[i, c + '_max_' + t] = tmp_past_values.max()
                                tmp.loc[i, c + '_min_' + t] = tmp_past_values.min()
                                tmp.loc[i, c + '_mean_' + t] = tmp_past_values.mean()
                                tmp.loc[i, c + '_median_' + t] = _tmp_past[c].median()
                                tmp.loc[i, c + '_var_' + t] = tmp_past_values.var()
                                tmp.loc[i, c + '_last_' + t] = tmp_past_values[-1]
                            tmp.loc[i, 'total_accuracy_' + t] = \
                                tmp.loc[i, 'num_correct_sum_' + t] / (
                                            tmp.loc[i, 'num_correct_sum_' + t] + tmp.loc[i, 'num_incorrect_sum_' + t])
                else:
                    past_days = days * 24 * 60 ** 2 * 10 ** 9
                    tmp_past = tmp[(tmp.loc[i, 'timestamp'] - past_days) < tmp.timestamp][
                        tmp.timestamp < tmp.loc[i, 'timestamp']]
                    if len(tmp_past) != 0:
                        for c in target_cols:
                            tmp_past_values = tmp_past[c].values
                            tmp.loc[i, c + f'_sum_{days}day'] = tmp_past_values.sum()
                            tmp.loc[i, c + f'_max_{days}day'] = tmp_past_values.max()
                            tmp.loc[i, c + f'_min_{days}day'] = tmp_past_values.min()
                            tmp.loc[i, c + f'_mean_{days}day'] = tmp_past_values.mean()
                            tmp.loc[i, c + f'_median_{days}day'] = tmp_past[c].median()
                            tmp.loc[i, c + f'_var_{days}day'] = tmp_past_values.var()
                            tmp.loc[i, c + f'_last_{days}day'] = tmp_past_values[-1]
                        tmp.loc[i, f'total_accuracy_{days}day'] = \
                            tmp.loc[i, f'num_correct_sum_{days}day'] / (
                                        tmp.loc[i, f'num_correct_sum_{days}day'] + tmp.loc[
                                    i, f'num_incorrect_sum_{days}day'])
                    # Bird Measurer, Cart Balancer, Cauldron Filler, Chest Sorter, and Mushroom Sorter
                    for t in title_cols:
                        _tmp_past = tmp_past[tmp_past.title == t]
                        if len(_tmp_past) != 0:
                            for c in target_cols:
                                tmp_past_values = _tmp_past[c].values
                                tmp.loc[i, c + f'_sum_{days}day_' + t] = tmp_past_values.sum()
                                tmp.loc[i, c + f'_max_{days}day_' + t] = tmp_past_values.max()
                                tmp.loc[i, c + f'_min_{days}day_' + t] = tmp_past_values.min()
                                tmp.loc[i, c + f'_mean_{days}day_' + t] = tmp_past_values.mean()
                                tmp.loc[i, c + f'_median_{days}day_' + t] = _tmp_past[c].median()
                                tmp.loc[i, c + f'_var_{days}day_' + t] = tmp_past_values.var()
                                tmp.loc[i, c + f'_last_{days}day_' + t] = tmp_past_values[-1]
                            tmp.loc[i, f'total_accuracy_{days}day_' + t] = \
                                tmp.loc[i, f'num_correct_sum_{days}day_' + t] / (
                                            tmp.loc[i, f'num_correct_sum_{days}day_' + t] + tmp.loc[
                                        i, f'num_incorrect_sum_{days}day_' + t])
        return tmp

    for (_, tmp) in df.groupby('installation_id'):

        tmp = tmp.sort_values('timestamp').reset_index(drop=True).reset_index()
        tmp = tmp.rename(columns={'index': 'count'})
        tmp = past_solved_feature(tmp, days=None)
        if DAYS:
            tmp = past_solved_feature(tmp, days=7)

        output = pd.concat([output, tmp])

    return output.reset_index(drop=True)


def clean_title_m(df):
    title_cols = ['Cart Balancer (Assessment)', 'Cauldron Filler (Assessment)',
                  'Mushroom Sorter (Assessment)', 'Chest Sorter (Assessment)', 'Bird Measurer (Assessment)']

    if DAYS:
        for title in title_cols:
            for c in ['num_correct', 'num_incorrect', 'accuracy_group']:
                for m in ['mean', 'max', 'min', 'median', 'sum', 'var', 'last']:
                    if f'{c}_{m}_{title}' not in df.columns:
                        df[f'{c}_{m}_{title}'] = np.nan
                    replace_index = df[df['title'] == title][df[f'{c}_{m}_{title}'].notnull()].index
                    df.loc[replace_index, f'{c}_title_{m}'] = df.loc[replace_index, f'{c}_{m}_{title}']
                    del df[f'{c}_{m}_{title}']
                    if f'{c}_{m}_7day_{title}' not in df.columns:
                        df[f'{c}_{m}_7day_{title}'] = np.nan
                    replace_index = df[df['title'] == title][df[f'{c}_{m}_7day_{title}'].notnull()].index
                    df.loc[replace_index, f'{c}_title_7day_{m}'] = df.loc[replace_index, f'{c}_{m}_7day_{title}']
                    del df[f'{c}_{m}_7day_{title}']
    else:
        for title in title_cols:
            for c in ['num_correct', 'num_incorrect', 'accuracy_group']:
                for m in ['mean', 'max', 'min', 'sum', 'var']:
                    if f'{c}_{m}_{title}' not in df.columns:
                        df[f'{c}_{m}_{title}'] = np.nan
                    replace_index = df[df['title'] == title][df[f'{c}_{m}_{title}'].notnull()].index
                    df.loc[replace_index, f'{c}_title_{m}'] = df.loc[replace_index, f'{c}_{m}_{title}']
                    del df[f'{c}_{m}_{title}']

    return df


def prepare_dataset(train_df, test_df, train_threshold, valid_threshold):
    with timer('Prepare input'):
        train_df, test_df = train_df.align(test_df, join='left', axis=1)
        train = pd.concat([test_df[test_df[TARGET].notnull()], train_df]).reset_index(drop=True)
        train = train.drop_duplicates(subset='game_session').reset_index(drop=True)
        train[TARGET] = train[TARGET].astype(int)
        test = test_df[test_df[TARGET].isnull()].reset_index(drop=True)
        with timer('train selection'):
            logger.info(f'train_threshold : {train_threshold}')
            train = train[train['count']>=train_threshold[0]][train['count']<=train_threshold[1]].reset_index(drop=True)
            folds = make_stratified_group_k_folds(train, ID, TARGET, GROUP, N_FOLD, seed=SEED)
            with timer('valid selection'):
                logger.info(f'valid_threshold : {valid_threshold}')
                valid_index = list(train[train['count']>=valid_threshold[0]][train['count']<=valid_threshold[1]].index)
                remove_index = set(train.index) - set(valid_index)
                folds.loc[remove_index, 'fold'] = -1
                print(folds['fold'].value_counts())
                folds.to_csv(f'train_threshold{train_threshold[1]}_valid_threshold{valid_threshold[1]}_folds.csv', index=None)
        target = train[TARGET]
        test = test[test['count']>=valid_threshold[0]][test['count']<=valid_threshold[1]].reset_index(drop=True)
        print(train.shape, folds.shape, target.shape, test.shape)
        missing_df = (test.isnull().sum() / len(test)).reset_index()
        missing_df.columns = ['Feature', 'missing_ratio']
        missing_features = missing_df[missing_df['missing_ratio']==1]['Feature'].values
        num_features = [c for c in test.columns if test.dtypes[c] != 'object']
        cat_features = ['title', 'world']
        features = num_features + cat_features
        drop_features = [ID, TARGET, 'accuracy', 'num_correct', 'num_incorrect', 'year', 'game_time',
                         'event_code', 'type', 'timestamp', 'event_count', 'count']
        features = [c for c in features if c not in drop_features]
        features = [c for c in features if c not in missing_features]
        ce_oe = ce.OrdinalEncoder(cols=cat_features, handle_unknown='impute')
        ce_oe.fit(train)
        train = ce_oe.transform(train)
        test = ce_oe.transform(test)
        
    return train, test, folds, features, cat_features, target


# ===========================================================
# main
# ===========================================================
def main():

    with timer('Data Loading'):
        train_df = load_df(path=df_path_dict['train'], df_name='train')
        test = load_df(path=df_path_dict['test'], df_name='test')
        test = reduce_mem_usage(test)
        # specs = load_df(path=df_path_dict['specs'], df_name='specs')
        sample_submission = load_df(path=df_path_dict['sample_submission'], df_name='sample_submission')

    with timer('Creating test_labels'):
        test = assessment(test)
        test, test_labels = create_test_labels(test, sample_submission)
        test_labels.to_csv('test_labels.csv', index=False)
        logger.info(f'test_labels shape : {test_labels.shape}')

    with timer('Time features'):
        test = extract_time_features(test)

    with timer('Extract user logs'):
        test_df = extract_user_logs(test, test_labels)
        del test, test_labels;
        gc.collect()
        logger.info(f'test_df shape : {test_df.shape}')
        
    with timer('Ratio features'):
        counter_cols = [c for c in test_df.columns if str(c).find('_Counter')>=0]
        test_df['sum_counter'] = test_df[counter_cols].sum(axis=1)
        for c in counter_cols:
            test_df[f'Ratio_{c}'] = test_df[c] / test_df['sum_counter']

    with timer('Past solved features'):
        test_df = past_solved_features(test_df)
        test_df = clean_title_m(test_df)
        test_df.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in test_df.columns]
        test_df = test_df.sort_index(axis='columns')
        # test_df = reduce_mem_usage(test_df)
        logger.info(f'test_df shape : {test_df.shape}')
        test_df.to_pickle('test.pkl')

    with timer('Training for 1st assesment = train:[0, 3], valid:[0, 0]'):
        with timer('Prepare input'):
            train, test, folds, features, cat_features, target = prepare_dataset(train_df, test_df, [0, 3], [0, 0])
        with timer('Run pipeline'):
            lgb_param = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,
                'data_random_seed': SEED,
                'num_leaves': 25,
                'subsample': 0.7,
                'subsample_freq': 2,
                'max_depth': 6,
                'reg_alpha': 0.1,
                'colsample_bytree': 0.7,
                'min_split_gain': 0.5,
                'reg_lambda': 0.1,
                'min_data_in_leaf': 100,
                'verbosity': -1,
            }

            xgb_param = {
                'booster': "gbtree",
                'eval_metric': 'rmse',
                'eta': 0.02,
                'max_depth': 6,
                'min_child_weight': 90,
                'gamma': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.8,
                'alpha': 1,
                'lambda': 1,
                'seed': SEED,
                'silent': 1
            }

            cat_param = {
                'loss_function': 'RMSE',
                'eval_metric': 'RMSE',
                'bootstrap_type': 'Bayesian',
                'iterations': 10000,
                'learning_rate': 0.03,
                'max_depth': 6,
                'random_seed': SEED,
                'bagging_temperature': 0.9,
                'l2_leaf_reg': 0.9,
                'random_strength': 1,
                'od_type': 'Iter',
                'metric_period': 1000,
                'od_wait': 100,
            }

            nn_param = {
                'batch_size': 256,
                'epochs': 20,
                'lr': 0.1,
                'emb_drop': 0.5,
            }
            
            model_pipeline(train, test, folds, features, cat_features, target, 
                           lgb_param, xgb_param, cat_param, nn_param, version=1, submission_file='submission1.csv', DROP=False)
        
    with timer('Training for 2nd~9th assesment = train:[1, 8], valid:[1, 8]'):
        with timer('Prepare input'):
            train, test, folds, features, cat_features, target = prepare_dataset(train_df, test_df, [1, 8], [1, 8])
        with timer('Run pipeline'):
            lgb_param = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,
                'data_random_seed': SEED,
                'num_leaves': 25,
                'subsample': 0.7,
                'subsample_freq': 2,
                'max_depth': 6,
                'reg_alpha': 0.1,
                'colsample_bytree': 0.7,
                'min_split_gain': 0.5,
                'reg_lambda': 0.1,
                'min_data_in_leaf': 100,
                'verbosity': -1,
            }

            xgb_param = {
                'booster': "gbtree",
                'eval_metric': 'rmse',
                'eta': 0.02,
                'max_depth': 7,
                'min_child_weight': 120,
                'gamma': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.7,
                'alpha': 0.1,
                'lambda': 0.1,
                'seed': SEED,
                'silent': 1
            }

            cat_param = {
                'loss_function': 'RMSE',
                'eval_metric': 'RMSE',
                'bootstrap_type': 'Bayesian',
                'iterations': 10000,
                'learning_rate': 0.03,
                'max_depth': 6,
                'random_seed': SEED,
                'bagging_temperature': 0.8,
                'l2_leaf_reg': 1,
                'random_strength': 1,
                'od_type': 'Iter',
                'metric_period': 1000,
                'od_wait': 100,
            }

            nn_param = {
                'batch_size': 256,
                'epochs': 20,
                'lr': 0.1,
                'emb_drop': 0.5,
            }
            
            model_pipeline(train, test, folds, features, cat_features, target, 
                           lgb_param, xgb_param, cat_param, nn_param, version=2, submission_file='submission2.csv', DROP=False)

    with timer('Training for 10th~ assesment = train:[2, 100000000], valid:[9, 100000000]'):
        with timer('Prepare input'):
            train, test, folds, features, cat_features, target = prepare_dataset(train_df, test_df, [2, 100000000], [9, 100000000])
        with timer('Run pipeline'):
            lgb_param = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': 0.01,
                'data_random_seed': SEED,
                'num_leaves': 25,
                'subsample': 0.7,
                'subsample_freq': 2,
                'max_depth': 6,
                'reg_alpha': 0.1,
                'colsample_bytree': 0.7,
                'min_split_gain': 0.5,
                'reg_lambda': 0.1,
                'min_data_in_leaf': 100,
                'verbosity': -1,
            }

            xgb_param = {
                'booster': "gbtree",
                'eval_metric': 'rmse',
                'eta': 0.02,
                'max_depth': 5,
                'min_child_weight': 100,
                'gamma': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.8,
                'alpha': 1,
                'lambda': 1,
                'seed': SEED,
                'silent': 1
            }

            cat_param = {
                'loss_function': 'RMSE',
                'eval_metric': 'RMSE',
                'bootstrap_type': 'Bayesian',
                'iterations': 10000,
                'learning_rate': 0.03,
                'max_depth': 7,
                'random_seed': SEED,
                'bagging_temperature': 0.9,
                'l2_leaf_reg': 1,
                'random_strength': 1,
                'od_type': 'Iter',
                'metric_period': 1000,
                'od_wait': 100,
            }

            nn_param = {
                'batch_size': 256,
                'epochs': 20,
                'lr': 0.1,
                'emb_drop': 0.5,
            }
            
            model_pipeline(train, test, folds, features, cat_features, target, 
                           lgb_param, xgb_param, cat_param, nn_param, version=3, submission_file='submission3.csv', DROP=False)
        
    with timer('Prepare submission'):
        sub1 = pd.read_csv('submission1.csv')
        sub2 = pd.read_csv('submission2.csv')
        sub3 = pd.read_csv('submission3.csv')
        subs = pd.concat([sub1, sub2, sub3]).reset_index(drop=True)
        print(subs.shape)
        submission = pd.concat([sample_submission.set_index(ID).drop(columns=[TARGET]), subs.set_index(ID)], axis=1).reset_index()
        if 'index' in submission.columns:
            submission = submission.rename(columns={'index': ID})
        submission.to_csv('submission.csv', index=False)
    
    with timer('Truncated CV'):
        cols = [ID, 'game_session', TARGET, 'count']
        train = pd.concat([test_df[test_df[TARGET].notnull()][cols], train_df[cols]]).reset_index(drop=True)
        train = train.drop_duplicates(subset='game_session').reset_index(drop=True)
        train[TARGET] = train[TARGET].astype(int)
        oof_df1 = pd.read_csv(OUTPUT_DICT + f'oof_averaging_soft1.csv')
        oof_df2 = pd.read_csv(OUTPUT_DICT + f'oof_averaging_soft2.csv')
        oof_df3 = pd.read_csv(OUTPUT_DICT + f'oof_averaging_soft3.csv')
        oof_df = pd.concat([oof_df1, oof_df2, oof_df3]).reset_index(drop=True)
        oof_df.columns = ['game_session', 'oof_soft']
        train = train.merge(oof_df, on='game_session')
        logger.info("CV score: {:<8.5f}"
                    .format(np.sqrt(metrics.mean_squared_error(train[TARGET].values, train['oof_soft'].values))))
        oof_df1 = pd.read_csv(OUTPUT_DICT + f'oof_averaging_hard1.csv')
        oof_df2 = pd.read_csv(OUTPUT_DICT + f'oof_averaging_hard2.csv')
        oof_df3 = pd.read_csv(OUTPUT_DICT + f'oof_averaging_hard3.csv')
        oof_df = pd.concat([oof_df1, oof_df2, oof_df3]).reset_index(drop=True)
        oof_df.columns = ['game_session', 'oof_hard']
        train = train.merge(oof_df, on='game_session')
        logger.info("CV QWK score: {:<8.5f}"
                .format(quadratic_weighted_kappa(train['oof_hard'].values, train[TARGET].values)))
        eval_with_truncated_data(train['oof_hard'], train[TARGET], train[ID].values, n_trials=100)


if __name__ == "__main__":
    main()
