import numpy as np
import pandas as pd


def mutate(df: pd.DataFrame) -> None:
    v = df.values
    v = np.zeros(3)
    v -= 1
