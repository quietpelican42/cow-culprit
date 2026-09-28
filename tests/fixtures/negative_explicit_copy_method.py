import pandas as pd


def mutate(df: pd.DataFrame) -> None:
    v = df.values.copy()
    v -= 1


def reassigned_after_copy(df: pd.DataFrame) -> None:
    v = df.values
    v = v.copy()
    v -= 1
