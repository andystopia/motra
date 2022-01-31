import numpy as np
import pandas as pd


def distance(x0: pd.Series, y0: pd.Series, x1: pd.Series, y1: pd.Series):
    return np.sqrt(np.power(x1 - x0, 2) + np.power(y1 - y0, 2))
