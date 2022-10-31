from IPython.display import display

import pandas as pd


def basic_info(df):
    """ 基礎情報"""
    print("データ数", df.shape)
    print("="*50)

    print("カラム", df.columns.values)
    print("="*50)

    print("dfの上下5行")
    display(df.head())
    print("="*50)

    print("カラム別データ数と型")
    display(df.info())
    print("="*50)

    print("統計情報")
    display(df.describe())


def groupby_mean(df, target_column, group_columns):
    df = df[group_columns].groupby([target_column], as_index=False).mean()\
                          .sort_values(by='Survived', ascending=False)
    return df
