import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from typing import Union

from .util import distance, convert_to_relative_unit
from .constants import FPS


def stats(flies_data: pd.DataFrame, time_interval: float = 1) -> pd.DataFrame:
    """ Generates average distance and velocity within each time interval for each fly.

    Parameters
    ----------
    flies_data: pd.DataFrame
        Coordinates of flies. 
        Flies can be in the same or different arenas. Must have 'pos x', 'pos y', 
        'fly_id', and 'timestamp' columns.

    time_frames: int
        Time interval (unit: seconds).

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame of average distance and velocity for each time interval.
        'timestamp' column denotes the starting timestamp of the time interval.
    """

    # Calculates distance each fly made between current frame and previous frame.
    coordinates = flies_data[["pos x", "pos y", "fly_id", "timestamp"]].copy()
    coordinates[["pos_x_lagged", "pos_y_lagged"]
                ] = coordinates.groupby("fly_id")[["pos x", "pos y"]].shift(1)
    coordinates["distance"] = distance(
        coordinates["pos x"], coordinates["pos y"],
        coordinates["pos_x_lagged"], coordinates["pos_y_lagged"]
    )

    # Calculates total distance each fly travels within a time interval.
    coordinates["distance"] = coordinates.groupby("fly_id")[
        "distance"].cumsum()
    coordinates["index_in_group"] = coordinates.groupby("fly_id").cumcount()
    frames = time_interval / (1 / FPS)
    coordinates["mod_of_index"] = coordinates["index_in_group"] % frames
    coordinates = coordinates.loc[coordinates["mod_of_index"] == 0]
    coordinates.loc[coordinates["index_in_group"] == 0, "distance"] = 0

    coordinates["distance"] = coordinates["distance"] - \
        coordinates.groupby("fly_id")["distance"].shift(1)

    # Retrieves columns of interest and calculates velocity
    result = coordinates[["fly_id", "timestamp", "distance"]].copy()
    result["velocity (per second)"] = result["distance"] / time_interval

    return result.reset_index(drop=True)


def time_distribution_by_quadrant(coordinates: pd.DataFrame,
                                  arena_center: tuple[float]) -> pd.DataFrame:
    """ Calculates time distribution of each fly across four quadrants of one arena.

    Parameters
    ----------
    coordinates: pd.DataFrame
        Coordinates of every fly in the arena.
        Must have 'pos x', 'pos y', 'fly_id', and 'timestamp' columns.

    arena_center: tuple[float]
        Coordinates of the center of the arena.

    Returns
    -------
    pd.DataFrame
        Pivot table showing the time distribution of each fly across quadrants.
    """

    center_x = arena_center[0]
    center_y = arena_center[1]

    _coords = coordinates.copy()
    _coords["relative_pos_x"] = _coords["pos x"] - center_x
    _coords["relative_pos_y"] = _coords["pos y"] - center_y

    # Matches coordinates with quadrant
    conditions = [
        np.logical_and(np.greater_equal(
            _coords["relative_pos_x"], 0), np.greater_equal(_coords["relative_pos_y"], 0)),
        np.logical_and(np.less_equal(
            _coords["relative_pos_x"], 0), np.greater_equal(_coords["relative_pos_y"], 0)),
        np.logical_and(np.less_equal(
            _coords["relative_pos_x"], 0), np.less_equal(_coords["relative_pos_y"], 0)),
        np.logical_and(np.greater_equal(
            _coords["relative_pos_x"], 0), np.less_equal(_coords["relative_pos_y"], 0)),
    ]
    outputs = [1, 2, 3, 4]
    _coords["quadrant"] = np.select(conditions, outputs)

    # Finds number of times the fly appears in a quadrant, for every fly
    fly_quadrant_dist = _coords.groupby(
        ["fly_id", "quadrant"])["pos x"].count().reset_index()
    fly_quadrant_dist_pivot = fly_quadrant_dist.pivot(
        index="fly_id", columns="quadrant", values="pos x")
    fly_quadrant_dist_pivot.fillna(0, inplace=True)

    # Normalizes the distribution by using percentages
    total_time_by_fly = fly_quadrant_dist_pivot.sum(axis=1)
    fly_quadrant_dist_pivot = fly_quadrant_dist_pivot.apply(
        lambda row: row / total_time_by_fly)

    return fly_quadrant_dist_pivot


def time_dist_circle(coordinates_dfs: list[pd.DataFrame],
                     centers: list[tuple[float]],
                     arena_radius: float,
                     aoi_circle_radii: list[float],
                     labels: list[str] = None) -> pd.DataFrame:
    """ Calculates total time distribution of flies across concentric rings of one arena.
    This function can be applied to one or multiple arenas. All arena must have the same radius.

    Parameters
    ----------
    coordinates_dfs: list[pd.DataFrame]
        List of coordinates DataFrames. 
        Must have 'pos x', 'pos y', 'fly_id', and 'timestamp' columns.

    centers: list[tuple[float]]
        Coordinates of the centers of each arena.

    arena_radius: float
        Relative radius of the arenas.

    aoi_circle_radii: list[float]
        Radii of the concentric rings.
        All radii must be smaller or equal to the radius of the arenas.

    labels: list[str]. Optional, default to None.
        Labels for each arena.

    Returns
    -------
    pd.DataFrame
        Pivot table showing the total time distribution of flies in rings 
        between concentric circles of different radii.
    """

    # Validate that radii of concentric rings ares smaller than the radius of arenas.
    invalid_radii = [
        r for r in aoi_circle_radii if convert_to_relative_unit(r, arena_radius) > arena_radius]
    if len(invalid_radii) > 0:
        print("These radii are not valid because they are larger than the arena's radius")
        print(invalid_radii)
        return pd.DataFrame()

    # Convert radii into unit used by tracking software.
    aoi_radii_rel_unit = sorted([convert_to_relative_unit(
        r, arena_radius) for r in aoi_circle_radii])

    # Create hit test dictionary that shows the distribution of coordinates across rings.
    hit_test = dict()
    for idx in range(len(aoi_radii_rel_unit)):
        for coord, center in zip(coordinates_dfs, centers):
            coordinates = coord.copy()
            coordinates["distance_from_center"] = distance(
                coordinates["pos x"], coordinates["pos y"], center[0], center[1])

            # Check which coordinates fall into the current ring.
            # The coordinate is labeled 1 if it falls into the ring, 0 otherwise.
            if idx > 0:
                coordinates["hit_test"] = np.where(
                    np.logical_and(coordinates["distance_from_center"] <= aoi_radii_rel_unit[idx],
                                   coordinates["distance_from_center"] > aoi_radii_rel_unit[idx - 1]), 1, 0)
            else:
                coordinates["hit_test"] = np.where(
                    coordinates["distance_from_center"] <= aoi_radii_rel_unit[idx], 1, 0)

            # Calculate percentage of coordinates falling into the ring.
            hit_percent = coordinates["hit_test"].sum() / coordinates.shape[0]

            # Concatenate data to the hit test dictionary.
            radius_mm_key = str(aoi_circle_radii[idx])
            if radius_mm_key not in hit_test.keys():
                hit_test[radius_mm_key] = [hit_percent]
            else:
                hit_test[radius_mm_key].append(hit_percent)

    # Convert the hit test dictionary into dataframe and return.
    circle_dist = pd.DataFrame(hit_test)
    circle_dist.columns = ["Circle of radius {}mm".format(
        r) for r in circle_dist.columns]
    if labels is not None:
        circle_dist.index = labels

    return circle_dist


def visualize_stats(stats: pd.DataFrame, figsize: tuple[int] = (15, 20),
                    stats_columns: list[str] = [
                        "distance (mm)", "velocity (mm per second)"],
                    ylims: list[float] = [None, None],
                    x_label_freq: int = 1,
                    ylabels: list[str] = [
                        "distance (mm)", "velocity (mm per second)"]) -> None:
    """ Visualizes average distance and velocity  within each time interval of each fly with bar chart.

    Parameters
    ----------
    stats: pd.DataFrame
        Pandas DataFrame of average distance and velocity for each time interval.
        Must have 'fly_id' and 'timestamp' columns. 'timestamp' column denotes 
        the starting timestamp of the time interval.

    figsize: tuple[int]. Optional, default to (15, 20).
        Size of the figure. 
        This argument will be passed as the figsize argument of matplotlib.pyplot.subplots.

    stats_columns: list[str]. Optional, default to ["distance (mm)", "velocity (mm per second)"].
        Names of the columns that need to be visualized.

    ylims: list[float]. Optional, default to [None, None].
        Vertical limits of each statistics graph.
        The length of ylims must match that of stats_column.

    x_label_freq: int. Optional, default to None.
        Frequency of the horizontal-axis timestamp label.

    ylabels: list[str]. Optional, default to ["distance (mm)", "velocity (mm per second)"].
        Labels for the vertical axis each statistics graph.
        The length of ylabels must match that of stats_column.

    Returns
    -------
    None
    """

    _, axes = plt.subplots(nrows=len(stats_columns), figsize=figsize)

    for i in range(len(stats_columns)):
        sns.barplot(data=stats, x="timestamp",
                    y=stats_columns[i], hue="fly_id", ax=axes[i], palette="tab10")

        for idx, t in enumerate(axes[i].get_xticklabels()):
            if (idx % (stats.shape[0] // x_label_freq)) != 0:
                t.set_visible(False)

        axes[i].tick_params(bottom=False)
        axes[i].set_xticklabels(axes[i].get_xticks(), rotation=90)
        axes[i].set_ylim(top=ylims[i])
        axes[i].set_ylabel(ylabels[i])

    plt.show()


def graph_time_distribution_by_quadrant(
        fly_quadrant_dist_pivot: pd.DataFrame,
        cmap: Union[colors.Colormap, str] = "Blues",
        figsize: tuple[int] = (10, 10)) -> None:
    """ Visualizes the time distribution of each fly across four quadrants with heatmap.

    Parameters
    ----------
    fly_quadrant_dist_pivot: pd.DataFrame
        Pivot table showing the time distribution of each fly in each quadrant.
        Should be the pivot table returned from time_distribution_by_quadrant.

    cmap: matplotlib.colors.Colormap or str. Optional, default to 'viridis'.
        Color palette.
        This argument will be passed as the cmap argument of seaborn.heatmap.
        Options: https://matplotlib.org/stable/tutorials/colors/colormaps.html.

    figsize: tuple[int]. Optional, default to (10, 10).
        Size of the figure. 
        This argument will be passed as the figsize argument of matplotlib.pyplot.subplots.

    Returns
    -------
    None
    """

    plt.figure(figsize=figsize)
    sns.heatmap(fly_quadrant_dist_pivot, annot=True, fmt=".2f", cmap=cmap)
    plt.show()


def graph_time_dist_circle(circle_dist: pd.DataFrame, cmap: str = "Blues",
                           figsize: tuple = (10, 10)) -> None:

    plt.show()
