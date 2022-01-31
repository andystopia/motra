import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from motra.util import distance


def stats(flies_data: pd.DataFrame, frames: int = 30) -> pd.DataFrame:
    """
    Generates statistics (distance, velocity, etc.) based on number of frames.
    Parameters
    ----------
    coordinates: pd.DataFrame

    frames: int
        number of frames
    """
    result = pd.DataFrame()

    coordinates = flies_data[["pos x", "pos y", "fly_id", "timestamp"]].copy()
    coordinates[["pos_x_lagged", "pos_y_lagged"]
                ] = coordinates.groupby(["fly_id", coordinates.index // frames])[["pos x", "pos y"]].shift(1)

    coordinates["distance"] = distance(
        coordinates["pos x"], coordinates["pos y"], coordinates["pos_x_lagged"], coordinates["pos_y_lagged"])

    result = coordinates.groupby(
        ["fly_id", coordinates.index // frames]).agg(
            timestamp_start=("timestamp", "first"),
            timestamp_end=("timestamp", "last"),
            distance=("distance", "sum"),
    )

    result["velocity (per second)"] = result["distance"] / \
        (result["timestamp_end"] - result["timestamp_start"])

    return result.reset_index().drop("level_1", axis=1)


def visualize_stats(stats: pd.DataFrame, figsize: tuple = (15, 10),
                    columns: list = ["distance", "velocity (per second)"]) -> None:

    stats_copy = stats.copy()
    stats_copy["timestamp"] = stats_copy["timestamp_start"].astype(
        str) + " - " + stats_copy["timestamp_end"].astype(str)

    _, axes = plt.subplots(nrows=len(columns), figsize=figsize)

    for i in range(len(columns)):
        sns.lineplot(data=stats_copy, x="timestamp_start",
                     y=columns[i], hue="fly_id", ax=axes[i])
        axes[i].set_xticklabels(axes[i].get_xticks(), rotation=45)

    plt.show()

    return stats_copy
