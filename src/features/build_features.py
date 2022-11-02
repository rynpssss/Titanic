import pandas as pd


def create_title(
    df: pd.DataFrame, 
    replace_title: dict,
    column: str = "title",
):
    """ 性別敬称取得
    
        Mr./Ms.といった[.]で終わる名称に対して正規表現で取得
    """

    df[column] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    for replace, raw in replace_title.items():
        df[column] = df[column].replace(raw, replace)
    return df


def create_mapping(
    df: pd.DataFrame,
    mapping: dict,
    column: str = "title",
):
    """ カテゴリ変数のマッピング

        one-hot-encoding等の方が良いが、簡単な場合は使用
    """

    df[column] = df[column].map(mapping)
    df[column] = df[column].fillna(0)

    return df


class ModelDF():
    def __init__(self, df, funcs: dict):
        self.df = df
        self.funcs = funcs

    def __len__(self):
        return self.df.shape[0]

    def create_feature(self):
        for func, f_args in self.funcs.items():
            self.df = func(self.df, *f_args)

    def create_y_column(self, column: str):
        self.y = self.df[column]
        self.df = self.df.drop(column, axis=1)