import pandas as pd


def update(df: pd.DataFrame) -> None:
    df.loc[:, "a"] -= 1


def plain_numpy(x: "object") -> None:
    import numpy as np

    arr = np.zeros(5)
    arr -= 1
    arr[0] = 3
    arr.sort()

    literal = np.asarray([1, 2, 3])
    literal -= 1
