import pandas as pd


def delete_duplicated_columns(df: pd.DataFrame):
    df = df.loc[:, ~df.columns.duplicated()]
    return df
