import numpy as np
import pandas as pd


def mutate(s: pd.Series) -> None:
    values = s.array
    np.subtract(values, 1, out=values)
