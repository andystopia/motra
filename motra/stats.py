import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from motra.util import distance


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
    frames = time_frame / (1 / 30)
    coordinates["mod_of_index"] = coordinates["index_in_group"] % frames
    coordinates = coordinates.loc[coordinates["mod_of_index"] == 0]
    coordinates.loc[coordinates["index_in_group"] == 0, "distance"] = 0

    coordinates["distance"] = coordinates["distance"] - \
        coordinates.groupby("fly_id")["distance"].shift(1)

    result = coordinates[["fly_id", "timestamp",
                          "pos x", "pos y", "distance"]].copy()
    result["velocity (per second)"] = result["distance"] / time_frame

    return result.reset_index(drop=True)


def visualize_stats(stats: pd.DataFrame, figsize: tuple = (15, 10),
                    columns: list = ["distance", "velocity (per second)"]) -> None:

    stats_copy = stats.copy()

    _, axes = plt.subplots(nrows=len(columns), figsize=figsize)

    for i in range(len(columns)):
        sns.lineplot(data=stats_copy, x="timestamp",
                     y=columns[i], hue="fly_id", ax=axes[i], palette="tab10")
        axes[i].set_xticklabels(axes[i].get_xticks(), rotation=90)

    plt.show()
