import numpy as np
import pandas as pd


def mutate(df: pd.DataFrame) -> None:
    arr = np.asarray(df)
    np.random.shuffle(arr)
