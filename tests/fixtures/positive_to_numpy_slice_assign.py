import pandas as pd


def mutate(s: pd.Series) -> None:
    arr = s.to_numpy()
    arr[0] = 0
