import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .util import distance
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

    # quadrant = (x0, y0, x1, y1)
    q1 = (center_x, center_y, center_x + arena_radius, center_y + arena_radius)
    q2 = (center_x - arena_radius, center_y, center_x, center_y + arena_radius)
    q3 = (center_x - arena_radius, center_y - arena_radius, center_x, center_y)
    q4 = (center_x, center_y - arena_radius, center_x + arena_radius, center_y)
    quadrants = (q1, q2, q3, q4)

    def _hit_test(x: float, y: float) -> int:

        for idx, quad in enumerate(quadrants):
            x0, y0, x1, y1 = quad
            if x0 <= x <= x1 and y0 <= y <= y1:
                return idx

    coordinates_copy = coordinates.copy()
    coordinates_copy["quadrant"] = coordinates_copy.apply(
        lambda row: _hit_test(row["pos x"], row["pos y"]), axis=1)

    fly_quadrant_dist = coordinates_copy.groupby(
        ["fly_id", "quadrant"])["pos x"].count().reset_index()
    fly_quadrant_dist_pivot = fly_quadrant_dist.pivot(
        index="fly_id", columns="quadrant", values="pos x")
    fly_quadrant_dist_pivot.fillna(0, inplace=True)
    total_time_by_fly = fly_quadrant_dist_pivot.sum(axis=1)
    fly_quadrant_dist_pivot = fly_quadrant_dist_pivot.apply(
        lambda row: row / total_time_by_fly)

    return fly_quadrant_dist_pivot


def visualize_stats(stats: pd.DataFrame, figsize: tuple = (15, 20),
                    columns: list = ["distance", "velocity (per second)"],  ylims: int = [None, None],
                    x_label_freq: int = 1) -> None:
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

    plt.show()


def graph_time_distribution_by_quadrant(fly_quadrant_dist_pivot: pd.DataFrame,
                                        cmap: str = "Blues", figsize: tuple = (10, 10)) -> None:

    plt.figure(figsize=figsize)
    sns.heatmap(fly_quadrant_dist_pivot, annot=True, fmt=".2f", cmap=cmap)
    plt.show()
