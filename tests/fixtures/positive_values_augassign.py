import pandas as pd


def mutate(df: pd.DataFrame) -> None:
    v = df.values
    v -= 1
