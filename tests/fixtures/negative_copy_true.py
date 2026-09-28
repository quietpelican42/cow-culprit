import pandas as pd


def mutate(df: pd.DataFrame) -> None:
    v = df.to_numpy(copy=True)
    v -= 1
