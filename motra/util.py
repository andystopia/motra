import numpy as np
import pandas as pd
from typing import Union

from .smallestenclosingcircle import make_circle

from .constants import FPS, RADIUS


def distance(x0: Union[pd.Series, float],
             y0: Union[pd.Series, float],
             x1: Union[pd.Series, float],
             y1: Union[pd.Series, float]) -> Union[np.ndarray, float]:
    """ Calculates Euclidean distance between two points.

    Parameters
    ----------
    x0: pd.Series, or float
        x-coordinate of first point.

    y0: pd.Series, or float
        y-coordinate of first point.

    x1: pd.Series, or float
        x-coordinate of second point.

    y1: pd.Series, or float
        y-coordinate of first point.

    Returns
    -------
    np.ndarray, or float
        Distance between two input points.
    """

    return np.sqrt(np.power(x1 - x0, 2) + np.power(y1 - y0, 2))


def sample_by_fly(flies_data: pd.DataFrame, seconds: float = 10) -> pd.DataFrame:
    """ Returns the first seconds of each fly's coordinates.

    Parameters
    ----------
    flies_data: pd.DataFrame
        Coordinates of flies.
        Must have 'fly_id' column.

    seconds: float. Optional, default to 10.
        Number of seconds to sample.

    Returns
    -------
    pd.DataFrame.
        Coordinates of flies in the first seconds.
    """

    if seconds is None:
        return flies_data

    return flies_data.groupby("fly_id").head(seconds * FPS + 1)


def convert_to_mm(
        original_distance: Union[pd.Series, np.ndarray, float],
        radius: float = RADIUS) -> Union[pd.Series, np.ndarray, float]:
    """ Converts relative unit of tracking software to milimeters.

    Parameters
    ----------
    original_distance: pd.Series, np.ndarray, or float
        Distance in relative unit of tracking software.

    radius: float.
        Radius of the arena in mm.

    Returns
    -------
    pd.Series, np.ndarray, or float
        Distance in mm.
    """

    scale_factor = RADIUS / radius
    return original_distance * scale_factor


def convert_to_relative_unit(
        distance_in_mm: Union[pd.Series, np.array, float],
        radius: float = RADIUS) -> Union[pd.Series, np.array, float]:
    """ Converts mm to relative unit of tracking software.

    Parameters
    ----------
    distance_in_mm: pd.Series, np.ndarray, or float
        Distance in mm.

    radius: float.
        Radius of the arena in mm.

    Returns
    -------
    pd.Series, np.ndarray, or float
        Distance in relative unit of tracking software.
    """

    scale_factor = radius / RADIUS
    return distance_in_mm * scale_factor


def get_center_radius(flies_data: pd.DataFrame) -> tuple[tuple[float], float]:
    """ Returns the center and radius of the arena.

    Parameters
    ----------
    flies_data: pd.DataFrame
        Coordinates of flies from one arena.
        Must have 'pos_x', 'pos_y' columns.

    Returns
    -------
    tuple[float], float
        Coordinate of center and radius of the arena..
    """

    coordinates = flies_data[["pos x", "pos y"]].copy().to_numpy()
    center_x, center_y, radius = make_circle(coordinates)
    return (center_x, center_y), radius
