import pandas as pd


def create_title(df: pd.DataFrame, replace_title: dict):
    """ 性別敬称取得
    
        Mr./Ms.といった[.]で終わる名称に対して正規表現で取得
    """
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    for replace, raw in replace_title.items():
        df["Title"] = df["Title"].replace(raw, replace)
    return df


def create_mapping(df: pd.DataFrame, mapping: dict):
    """ カテゴリ変数のマッピング
    
        one-hot-encoding等の方が良いが、簡単な場合は使用
    """
    df['Title'] = df['Title'].map(mapping)
    df['Title'] = df['Title'].fillna(0)

    return df