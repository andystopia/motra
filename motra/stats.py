import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .util import distance, convert_to_relative_unit
from .constants import FPS


def stats(flies_data: pd.DataFrame, time_frame: float = 1) -> pd.DataFrame:
    """
    Generates statistics (distance, velocity, etc.) based on number of frames.
    Parameters
    ----------
    coordinates: pd.DataFrame

    frames: int
        number of frames
    """

    # Calculates distance each fly made between current frame and previous frame.
    coordinates = flies_data[["pos x", "pos y", "fly_id", "timestamp"]].copy()
    coordinates[["pos_x_lagged", "pos_y_lagged"]
                ] = coordinates.groupby("fly_id")[["pos x", "pos y"]].shift(1)
    coordinates["distance"] = distance(
        coordinates["pos x"], coordinates["pos y"],
        coordinates["pos_x_lagged"], coordinates["pos_y_lagged"]
    )

    coordinates["distance"] = coordinates.groupby("fly_id")[
        "distance"].cumsum()
    coordinates["index_in_group"] = coordinates.groupby("fly_id").cumcount()
    frames = time_frame / (1 / FPS)
    coordinates["mod_of_index"] = coordinates["index_in_group"] % frames
    coordinates = coordinates.loc[coordinates["mod_of_index"] == 0]
    coordinates.loc[coordinates["index_in_group"] == 0, "distance"] = 0

    coordinates["distance"] = coordinates["distance"] - \
        coordinates.groupby("fly_id")["distance"].shift(1)

    result = coordinates[["fly_id", "timestamp",
                          "pos x", "pos y", "distance"]].copy()
    result["velocity (per second)"] = result["distance"] / time_frame

    return result.reset_index(drop=True)


def time_distribution_by_quadrant(coordinates: pd.DataFrame, arena_center: tuple[float, float],
                                  arena_radius: float) -> pd.DataFrame:

    center_x = arena_center[0]
    center_y = arena_center[1]

    _coords = coordinates.copy()
    _coords["absolute_pos_x"] = _coords["pos x"] - center_x
    _coords["absolute_pos_y"] = _coords["pos y"] - center_y

    conditions = [
        np.logical_and(np.greater_equal(
            _coords["absolute_pos_x"], 0), np.greater_equal(_coords["absolute_pos_y"], 0)),
        np.logical_and(np.less_equal(
            _coords["absolute_pos_x"], 0), np.greater_equal(_coords["absolute_pos_y"], 0)),
        np.logical_and(np.less_equal(
            _coords["absolute_pos_x"], 0), np.less_equal(_coords["absolute_pos_y"], 0)),
        np.logical_and(np.greater_equal(
            _coords["absolute_pos_x"], 0), np.less_equal(_coords["absolute_pos_y"], 0)),
    ]

    outputs = [1, 2, 3, 4]
    _coords["quadrant"] = np.select(conditions, outputs)

    fly_quadrant_dist = _coords.groupby(
        ["fly_id", "quadrant"])["pos x"].count().reset_index()
    fly_quadrant_dist_pivot = fly_quadrant_dist.pivot(
        index="fly_id", columns="quadrant", values="pos x")
    fly_quadrant_dist_pivot.fillna(0, inplace=True)
    total_time_by_fly = fly_quadrant_dist_pivot.sum(axis=1)
    fly_quadrant_dist_pivot = fly_quadrant_dist_pivot.apply(
        lambda row: row / total_time_by_fly)

    return fly_quadrant_dist_pivot


def time_dist_circle(coordinates_dfs: list[pd.DataFrame],
                     centers: list[tuple[float, float]],
                     arena_radius: float,
                     aoi_circle_radii: list[float],
                     labels: list[str] = None) -> pd.DataFrame:

    invalid_radii = [
        r for r in aoi_circle_radii if convert_to_relative_unit(r, arena_radius) > arena_radius]
    if len(invalid_radii) > 0:
        print("These radii are not valid because they are larger than the arena's radius")
        print(invalid_radii)
        return pd.DataFrame()

    aoi_radii_rel_unit = [convert_to_relative_unit(
        r, arena_radius) for r in aoi_circle_radii]

    hit_test = dict()
    for idx in range(len(aoi_radii_rel_unit)):
        for coord, center in zip(coordinates_dfs, centers):
            coordinates = coord.copy()
            coordinates["distance_from_center"] = distance(
                coordinates["pos x"], coordinates["pos y"], center[0], center[1])

            if idx > 0:
                coordinates["hit_test"] = np.where(
                    np.logical_and(coordinates["distance_from_center"] <= aoi_radii_rel_unit[idx],
                                   coordinates["distance_from_center"] > aoi_radii_rel_unit[idx - 1]), 1, 0)
            else:
                coordinates["hit_test"] = np.where(
                    coordinates["distance_from_center"] <= aoi_radii_rel_unit[idx], 1, 0)

            hit_percent = coordinates["hit_test"].sum() / coordinates.shape[0]

            radius_mm_key = str(aoi_circle_radii[idx])
            if radius_mm_key not in hit_test.keys():
                hit_test[radius_mm_key] = [hit_percent]
            else:
                hit_test[radius_mm_key].append(hit_percent)

    circle_dist = pd.DataFrame(hit_test)
    circle_dist.columns = ["Circle of radius {}mm".format(
        r) for r in circle_dist.columns]
    if labels is not None:
        circle_dist.index = labels

    return circle_dist


def visualize_stats(stats: pd.DataFrame, figsize: tuple = (15, 20),
                    columns: list[str] = [
                        "distance (mm)", "velocity (mm per second)"],
                    ylims: list[float, float] = [None, None],
                    x_label_freq: int = 1,
                    ylabels: list[str] = ["distance (mm)", "velocity (mm per second)"]) -> None:

    _, axes = plt.subplots(nrows=len(columns), figsize=figsize)

    for i in range(len(columns)):
        sns.barplot(data=stats, x="timestamp",
                    y=columns[i], hue="fly_id", ax=axes[i], palette="tab10")

        for idx, t in enumerate(axes[i].get_xticklabels()):
            if (idx % (stats.shape[0] // x_label_freq)) != 0:
                t.set_visible(False)

        axes[i].tick_params(bottom=False)
        axes[i].set_xticklabels(axes[i].get_xticks(), rotation=90)
        axes[i].set_ylim(top=ylims[i])
        axes[i].set_ylabel(ylabels[i])

    plt.show()


def graph_time_distribution_by_quadrant(fly_quadrant_dist_pivot: pd.DataFrame,
                                        cmap: str = "Blues", figsize: tuple = (10, 10)) -> None:

    plt.figure(figsize=figsize)
    sns.heatmap(fly_quadrant_dist_pivot, annot=True, fmt=".2f", cmap=cmap)
    plt.show()


def graph_time_dist_circle(circle_dist: pd.DataFrame, cmap: str = "Blues",
                           figsize: tuple = (10, 10)) -> None:

    plt.show()
