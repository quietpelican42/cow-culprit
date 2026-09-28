import numpy as np
import pandas as pd


def sort_in_place(df: pd.DataFrame) -> None:
    arr = df.values
    arr.sort()


def fill_in_place(s: pd.Series) -> None:
    arr = s.to_numpy()
    arr.fill(0)


def copyto_in_place(df: pd.DataFrame) -> None:
    dst = df.values
    np.copyto(dst, dst)
