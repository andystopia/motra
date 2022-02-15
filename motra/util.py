import numpy as np
import pandas as pd
from typing import Union

from .constants import FPS, RADIUS


def distance(x0: pd.Series, y0: pd.Series, x1: pd.Series, y1: pd.Series):
    return np.sqrt(np.power(x1 - x0, 2) + np.power(y1 - y0, 2))


def sample_by_fly(flies_data: pd.DataFrame, seconds: float = 10) -> pd.DataFrame:

    if seconds is None:
        return flies_data

    return flies_data.groupby("fly_id").head(seconds * FPS + 1)


def convert_to_mm(original_distance: Union[pd.Series, np.array, float],
                  radius: float) -> Union[pd.Series, np.array, float]:

    scale_factor = RADIUS / radius
    return original_distance * scale_factor
