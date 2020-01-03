import gc

import category_encoders as ce
import numpy as np
import pandas as pd

from collections import Counter
from typing import Set, Union
from pathlib import Path

from tqdm import tqdm

from .base import Feature
from src.utils.timer import timer

ID = "installation_id"
TARGET = "accuracy_group"

IoS = Union[int, str]


class NakamaV8(Feature):
    def create_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        input_dir = Path("../input/data-science-bowl-2019")
        if not input_dir.exists():
            input_dir = Path("input/data-science-bowl-2019")

        train_labels = pd.read_csv(input_dir / "train_labels.csv")
        sample_submission = pd.read_csv(input_dir / "sample_submission.csv")
        if "title_event_code" not in train_df.columns:
            train_df["title_event_code"] = list(
                map(lambda x, y: str(x) + "_" + str(y), train_df["title"],
                    train_df["event_code"]))
            test_df["title_event_code"] = list(
                map(lambda x, y: str(x) + "_" + str(y), test_df["title"],
                    test_df["event_code"]))

        with timer("creating test_labels"):
            test_df = assessment(test_df)
            test_df, test_labels = create_test_labels(test_df,
                                                      sample_submission)

        with timer("extract time features"):
            if "date" not in train_df.columns:
                train_df = extract_time_features(train_df)
                test_df = extract_time_features(test_df)
                print("skip extracting time features")
            else:
                train_df["timestamp"] = train_df["timestamp"].astype(int)
                test_df["timestamp"] = test_df["timestamp"].astype(int)

        with timer("Extract user logs"):
            train = extract_user_logs(train_df, train_labels)
            del train_df, train_labels
            gc.collect()

            test = extract_user_logs(test_df, test_labels)
            del test_df, test_labels
            gc.collect()

        with timer("Ratio features"):
            counter_cols = [
                c for c in train.columns if str(c).find("_Counter") >= 0
            ]
            train["sum_counter"] = train[counter_cols].sum(axis=1)
            for c in counter_cols:
                train[f"Ratio_{c}"] = train[c] / train["sum_counter"]
            counter_cols = [
                c for c in test.columns if str(c).find("_Counter") >= 0
            ]
            test["sum_counter"] = test[counter_cols].sum(axis=1)
            for c in counter_cols:
                test[f"Ratio_{c}"] = test[c] / test["sum_counter"]

        with timer("Past solved features"):
            train = past_solved_features(train)
            train = clean_title_m(train)
            train.columns = [
                "".join(c if c.isalnum() else "_" for c in str(x))
                for x in train.columns
            ]
            train = train.sort_index(axis='columns')

            test = past_solved_features(test)
            test = clean_title_m(test)
            test.columns = [
                "".join(c if c.isalnum() else "_" for c in str(x))
                for x in test.columns
            ]
            test = test.sort_index(axis='columns')

        train, test = train.align(test, join="left", axis=1)
        valid = test[test[TARGET].notnull()].reset_index(drop=True)
        test = test[test[TARGET].isnull()].reset_index(drop=True)

        train[TARGET] = train[TARGET].astype(int)
        valid[TARGET] = valid[TARGET].astype(int)

        num_features = [c for c in test.columns if test.dtypes[c] != 'object']
        cat_features = ['title', 'world']
        features = num_features + cat_features
        drop_features = [
            ID, TARGET, 'accuracy', 'num_correct', 'num_incorrect',
            'year', 'game_time', 'event_code', 'type',
            'timestamp', 'event_count']
        features = [c for c in features if c not in drop_features]
        ce_oe = ce.OrdinalEncoder(cols=cat_features, handle_unknown='impute')
        ce_oe.fit(train)

        train = train[features]
        valid = valid[features]
        test = test[features]

        train = ce_oe.transform(train)
        valid = ce_oe.transform(valid)
        test = ce_oe.transform(test)

        self.train = train
        self.valid = valid
        self.test = test


def extract_time_features(df: pd.DataFrame):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['year'] = df['timestamp'].dt.year
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['weekofyear'] = df['timestamp'].dt.weekofyear

    df['timestamp'] = df['timestamp'].astype(int)

    return df


def assessment(df: pd.DataFrame):

    df['num_correct'] = 0
    df['num_incorrect'] = 0
    df.loc[(((df.event_code == 4100) &
            (df.title != 'Bird Measurer (Assessment)')) &
            (df.type == 'Assessment')), 'num_correct'] = \
        df.loc[(((df.event_code == 4100) &
                (df.title != 'Bird Measurer (Assessment)')) &
                (df.type == 'Assessment'))]['event_data'].apply(
                    lambda x: x.find('"correct":true') >= 0) * 1
    df.loc[(((df.event_code == 4110) &
            (df.title == 'Bird Measurer (Assessment)')) &
            (df.type == 'Assessment')), 'num_correct'] = \
        df.loc[(((df.event_code == 4110) &
                (df.title == 'Bird Measurer (Assessment)')) &
                (df.type == 'Assessment'))]['event_data'].apply(
                    lambda x: x.find('"correct":true') >= 0) * 1
    df.loc[(((df.event_code == 4100) &
            (df.title != 'Bird Measurer (Assessment)')) &
            (df.type == 'Assessment')), 'num_incorrect'] = \
        df.loc[(((df.event_code == 4100) &
                (df.title != 'Bird Measurer (Assessment)')) &
                (df.type == 'Assessment'))]['event_data'].apply(
                    lambda x: x.find('"correct":false') >= 0) * 1
    df.loc[(((df.event_code == 4110) &
            (df.title == 'Bird Measurer (Assessment)')) &
            (df.type == 'Assessment')), 'num_incorrect'] = \
        df.loc[(((df.event_code == 4110) &
                (df.title == 'Bird Measurer (Assessment)')) &
                (df.type == 'Assessment'))]['event_data'].apply(
                    lambda x: x.find('"correct":false') >= 0) * 1

    return df


def create_test_labels(test: pd.DataFrame, sample_submission: pd.DataFrame):

    # assessment
    cols = [
        'installation_id', 'game_session', 'title', 'num_correct',
        'num_incorrect'
    ]
    test_labels = pd.concat(
        [
            test[((test.event_code == 4100) &
                  (test.title != 'Bird Measurer (Assessment)'))
                 & (test.type == 'Assessment')][cols].groupby(
                     ['installation_id', 'game_session', 'title'],
                     as_index=False).sum(),
            test[((test.event_code == 4110) &
                  (test.title == 'Bird Measurer (Assessment)'))
                 & (test.type == 'Assessment')][cols].groupby(
                     ['installation_id', 'game_session', 'title'],
                     as_index=False).sum()
        ])
    test_labels['accuracy'] = test_labels['num_correct'] / (
        test_labels['num_correct'] + test_labels['num_incorrect'])
    test_labels['accuracy_group'] = np.nan
    test_labels.loc[(test_labels['num_correct'] == 1) &
                    (test_labels['num_incorrect'] == 0), 'accuracy_group'] = 3
    test_labels.loc[(test_labels['num_correct'] == 1) &
                    (test_labels['num_incorrect'] == 1), 'accuracy_group'] = 2
    test_labels.loc[(test_labels['num_correct'] == 1) &
                    (test_labels['num_incorrect'] >= 2), 'accuracy_group'] = 1
    test_labels.loc[(test_labels['num_correct'] == 0), 'accuracy_group'] = 0
    test_labels['accuracy_group'] = test_labels['accuracy_group'].astype(int)

    # no assessment ( what we have to predict )
    key_cols = [ID, 'timestamp', 'event_code', 'type']
    last_assesment = test[test.event_code == 2000][
        test.type == 'Assessment'][key_cols].groupby(
            ID, as_index=False).max()
    last_assesment_df = last_assesment.merge(
        test[key_cols + ['game_session', 'title']], on=key_cols,
        how='left')[['installation_id', 'game_session', 'title']]

    # concat them
    test_labels = pd.concat([test_labels,
                             last_assesment_df]).reset_index(drop=True)

    # drop ['num_correct', 'num_incorrect'] after assessment
    test = test.drop(columns=['num_correct', 'num_incorrect']).reset_index(
        drop=True)

    return test, test_labels


def extract_user_logs(df: pd.DataFrame, df_labels: pd.DataFrame):

    logs = pd.DataFrame()
    nunique_cols = [
        'event_id', 'game_session', 'timestamp', 'event_data', 'event_count',
        'event_code', 'title', 'world', 'date', 'month', 'hour', 'dayofweek',
        'weekofyear', 'title_event_code'
    ]
    sum_cols = ['title_event_code', 'title', 'event_code', 'world', 'type']
    sum_values: Set[IoS] = set()
    for c in sum_cols:
        sum_values = sum_values | set(df[c].unique())

    def extract_user_log(tmp: pd.DataFrame, tmp_df: pd.DataFrame, days=None):

        sum_df = pd.DataFrame()
        if days is None:
            _sum_df = Counter({value: 0 for value in list(sum_values)})
            for i in range(len(tmp_df)):
                if i == 0:
                    tmp_past = tmp[tmp.timestamp < tmp_df.loc[i, 'timestamp']]
                else:
                    tmp_past = tmp[
                        tmp_df.loc[i - 1, 'timestamp'] <= tmp.timestamp][
                            tmp.timestamp < tmp_df.loc[i, 'timestamp']]
                if len(tmp_past) == 0:
                    sum_df = pd.concat(
                        [sum_df,
                         pd.DataFrame({
                             'No_playing_logs': [1]
                         })],
                        axis=0)
                else:
                    nunique_df = pd.DataFrame(
                        tmp[tmp.timestamp < tmp_df.loc[i, 'timestamp']]
                        [nunique_cols].nunique()).T.add_prefix('nunique_')
                    for c in sum_cols:
                        _sum_df.update(Counter(tmp_past[c].values))
                    concat_df = pd.concat([
                        nunique_df,
                        pd.DataFrame.from_dict(
                            _sum_df, orient='index').T.add_suffix('_Counter')
                    ],
                                          axis=1)
                    sum_df = pd.concat([sum_df, concat_df], axis=0)
        else:
            past_days = days * 24 * 60**2 * 10**9
            for i in range(len(tmp_df)):
                if i == 0:
                    tmp_past = tmp[
                        (tmp_df.loc[i, 'timestamp'] - past_days) < tmp.
                        timestamp][tmp.timestamp < tmp_df.loc[i, 'timestamp']]
                if len(tmp_past) == 0:
                    sum_df = pd.concat([
                        sum_df,
                        pd.DataFrame({
                            f'{days}day_No_playing_logs': [1]
                        })
                    ],
                                       axis=0)
                else:
                    nunique_df = pd.DataFrame(
                        tmp_past[nunique_cols].nunique()).T.add_prefix(
                            f'nunique_{days}day_')
                    _sum_df = Counter({value: 0 for value in list(sum_values)})
                    for c in sum_cols:
                        _sum_df.update(Counter(tmp_past[c].values))
                    concat_df = pd.concat([
                        nunique_df,
                        pd.DataFrame.from_dict(
                            _sum_df, orient='index').T.add_suffix('_Counter')
                    ],
                                          axis=1).add_prefix(f'{days}day_')
                    sum_df = pd.concat([sum_df, concat_df], axis=0)

        return sum_df

    for (_, tmp) in tqdm(
            df.groupby('installation_id'),
            total=df["installation_id"].nunique()):

        tmp = tmp.sort_values('timestamp').reset_index(drop=True)
        tmp_df = tmp[tmp.event_code == 2000][
            tmp.type == 'Assessment'].reset_index(drop=True)
        sum_df = extract_user_log(tmp, tmp_df, days=None)

        # concat
        _log = pd.concat([tmp_df, sum_df.reset_index(drop=True)], axis=1)
        logs = pd.concat([logs, _log], axis=0)

    not_merge_columns = ['installation_id', 'title']
    output = df_labels.merge(
        logs.drop(columns=not_merge_columns), on='game_session', how='left')

    return output.reset_index(drop=True)


def past_solved_features(df: pd.DataFrame):

    output = pd.DataFrame()
    target_cols = ['num_correct', 'num_incorrect', 'accuracy_group']
    title_cols = [
        'Cart Balancer (Assessment)', 'Cauldron Filler (Assessment)',
        'Mushroom Sorter (Assessment)', 'Chest Sorter (Assessment)',
        'Bird Measurer (Assessment)'
    ]

    def past_solved_feature(tmp, days=None):
        for i in range(len(tmp)):
            if i != 0:
                if days is None:
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
                                tmp.loc[i, 'num_correct_sum'] +
                                tmp.loc[i, 'num_incorrect_sum'])
                    for t in title_cols:
                        _tmp_past = tmp_past[tmp_past.title == t]
                        if len(_tmp_past) != 0:
                            for c in target_cols:
                                tmp_past_values = _tmp_past[c].values
                                tmp.loc[i, c + '_sum_' +
                                        t] = tmp_past_values.sum()
                                tmp.loc[i, c + '_max_' +
                                        t] = tmp_past_values.max()
                                tmp.loc[i, c + '_min_' +
                                        t] = tmp_past_values.min()
                                tmp.loc[i, c + '_mean_' +
                                        t] = tmp_past_values.mean()
                                tmp.loc[i, c + '_median_' +
                                        t] = _tmp_past[c].median()
                                tmp.loc[i, c + '_var_' +
                                        t] = tmp_past_values.var()
                                tmp.loc[i, c + '_last_' +
                                        t] = tmp_past_values[-1]
                            tmp.loc[i, 'total_accuracy_' + t] = \
                                tmp.loc[i, 'num_correct_sum_' + t] / (
                                    tmp.loc[i, 'num_correct_sum_' + t] +
                                    tmp.loc[i, 'num_incorrect_sum_' + t])
                else:
                    past_days = days * 24 * 60**2 * 10**9
                    tmp_past = tmp[(
                        tmp.loc[i, 'timestamp'] - past_days) < tmp.timestamp][
                            tmp.timestamp < tmp.loc[i, 'timestamp']]
                    if len(tmp_past) != 0:
                        for c in target_cols:
                            tmp_past_values = tmp_past[c].values
                            tmp.loc[i, c +
                                    f'_sum_{days}day'] = tmp_past_values.sum()
                            tmp.loc[i, c +
                                    f'_max_{days}day'] = tmp_past_values.max()
                            tmp.loc[i, c +
                                    f'_min_{days}day'] = tmp_past_values.min()
                            tmp.loc[i, c +
                                    f'_mean_{days}day'] = tmp_past_values.mean(
                                    )
                            tmp.loc[i, c +
                                    f'_median_{days}day'] = tmp_past[c].median(
                                    )
                            tmp.loc[i, c +
                                    f'_var_{days}day'] = tmp_past_values.var()
                            tmp.loc[i, c +
                                    f'_last_{days}day'] = tmp_past_values[-1]
                        tmp.loc[i, f'total_accuracy_{days}day'] = \
                            tmp.loc[i, f'num_correct_sum_{days}day'] / (
                                tmp.loc[i, f'num_correct_sum_{days}day'] +
                                tmp.loc[i, f'num_incorrect_sum_{days}day'])
                    for t in title_cols:
                        _tmp_past = tmp_past[tmp_past.title == t]
                        if len(_tmp_past) != 0:
                            for c in target_cols:
                                tmp_past_values = _tmp_past[c].values
                                tmp.loc[i, c + f'_sum_{days}day_' +
                                        t] = tmp_past_values.sum()
                                tmp.loc[i, c + f'_max_{days}day_' +
                                        t] = tmp_past_values.max()
                                tmp.loc[i, c + f'_min_{days}day_' +
                                        t] = tmp_past_values.min()
                                tmp.loc[i, c + f'_mean_{days}day_' +
                                        t] = tmp_past_values.mean()
                                tmp.loc[i, c + f'_median_{days}day_' +
                                        t] = _tmp_past[c].median()
                                tmp.loc[i, c + f'_var_{days}day_' +
                                        t] = tmp_past_values.var()
                                tmp.loc[i, c + f'_last_{days}day_' +
                                        t] = tmp_past_values[-1]
                            tmp.loc[i, f'total_accuracy_{days}day_' + t] = \
                                tmp.loc[
                                    i, f'num_correct_sum_{days}day_' + t
                                    ] / (
                                    tmp.loc[
                                        i, f'num_correct_sum_{days}day_' + t
                                        ] +
                                    tmp.loc[
                                        i, f'num_incorrect_sum_{days}day_' + t
                                        ])
        return tmp

    for (_, tmp) in tqdm(
            df.groupby('installation_id'),
            total=df["installation_id"].nunique()):

        tmp = tmp.sort_values('timestamp').reset_index(drop=True).reset_index()
        tmp = tmp.rename(columns={'index': 'count'})
        tmp = past_solved_feature(tmp, days=None)
        tmp = past_solved_feature(tmp, days=7)

        output = pd.concat([output, tmp])

    return output.reset_index(drop=True)


def clean_title_m(df: pd.DataFrame):

    title_cols = [
        'Cart Balancer (Assessment)', 'Cauldron Filler (Assessment)',
        'Mushroom Sorter (Assessment)', 'Chest Sorter (Assessment)',
        'Bird Measurer (Assessment)'
    ]

    for title in title_cols:
        for c in ['num_correct', 'num_incorrect', 'accuracy_group']:
            for m in ['mean', 'max', 'min', 'median', 'sum', 'var', 'last']:
                replace_index = df[df['title'] == title][
                    df[f'{c}_{m}_{title}'].notnull()].index
                df.loc[replace_index, f'{c}_title_{m}'] = df.loc[
                    replace_index, f'{c}_{m}_{title}']
                del df[f'{c}_{m}_{title}']
                replace_index = df[df['title'] == title][
                    df[f'{c}_{m}_7day_{title}'].notnull()].index
                df.loc[replace_index, f'{c}_title_7day_{m}'] = df.loc[
                    replace_index, f'{c}_{m}_7day_{title}']
                del df[f'{c}_{m}_7day_{title}']

    return df
