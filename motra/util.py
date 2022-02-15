import numpy as np
import pandas as pd
from typing import Union

from .smallestenclosingcircle import make_circle

from .constants import FPS, RADIUS


def distance(x0: Union[pd.Series, float], y0: Union[pd.Series, float],
             x1: Union[pd.Series, float], y1: Union[pd.Series, float]) -> Union[np.ndarray, float]:

    return np.sqrt(np.power(x1 - x0, 2) + np.power(y1 - y0, 2))


def sample_by_fly(flies_data: pd.DataFrame, seconds: float = 10) -> pd.DataFrame:

    if seconds is None:
        return flies_data

    return flies_data.groupby("fly_id").head(seconds * FPS + 1)


def convert_to_mm(original_distance: Union[pd.Series, np.array, float],
                  radius: float) -> Union[pd.Series, np.array, float]:

    scale_factor = RADIUS / radius
    return original_distance * scale_factor


def convert_to_relative_unit(distance_in_mm: Union[pd.Series, np.array, float],
                             radius: float) -> Union[pd.Series, np.array, float]:

    scale_factor = radius / RADIUS
    return distance_in_mm * scale_factor


def get_center_radius(fly_data: pd.DataFrame):

    coordinates = fly_data[["pos x", "pos y"]].copy().to_numpy()
    center_x, center_y, radius = make_circle(coordinates)
    return (center_x, center_y), radius
