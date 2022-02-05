import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd


def arena(coordinates: pd.DataFrame) -> tuple[tuple, int]:

    center = (0, 0)
    radius = 0

    min_x_idx = coordinates["pos x"].idxmin()
    max_x_idx = coordinates["pos x"].idxmax()
    min_y_idx = coordinates["pos y"].idxmin()
    max_y_idx = coordinates["pos y"].idxmax()

    diameter_x = coordinates.iloc[max_x_idx]["pos x"] - \
        coordinates.iloc[min_x_idx]["pos x"]
    diameter_y = coordinates.iloc[max_y_idx]["pos y"] - \
        coordinates.iloc[min_y_idx]["pos y"]

    if diameter_x < diameter_y:
        center_x = (coordinates.iloc[min_y_idx]["pos x"] +
                    coordinates.iloc[max_y_idx]["pos x"]) / 2
        center_y = (coordinates.iloc[min_y_idx]["pos y"] +
                    coordinates.iloc[max_y_idx]["pos y"]) / 2
        radius = diameter_y / 2
    else:
        center_x = (coordinates.iloc[min_x_idx]["pos x"] +
                    coordinates.iloc[max_x_idx]["pos x"]) / 2
        center_y = (coordinates.iloc[min_x_idx]["pos y"] +
                    coordinates.iloc[max_x_idx]["pos y"]) / 2
        radius = diameter_x / 2

    center = (center_x, center_y)

    return center, radius


def _arena_boundary(arena_center: tuple, arena_radius: int, figsize: int = 10):

    _, ax = plt.subplots(figsize=(figsize, figsize))

    arena_boundary = plt.Circle(
        arena_center, arena_radius, color="black", fill=False)
    ax.add_patch(arena_boundary)

    return ax


def arena_trajectory(coordinates: pd.DataFrame, arena_center: tuple,
                     arena_radius: int, figsize: int = 10) -> None:

    ax = _arena_boundary(arena_center, arena_radius, figsize)

    for fly_id in coordinates["fly_id"].unique():
        label = "fly {}".format(fly_id)
        ax.plot("pos x", "pos y",
                data=coordinates.loc[coordinates["fly_id"] == fly_id], label=label)

    ax.legend()
    plt.show()


def heatmap(coordinates: pd.DataFrame, arena_center: tuple,
            arena_radius: int, figsize: int = 10) -> None:

    ax = _arena_boundary(arena_center, arena_radius, figsize)

    sns.kdeplot(data=coordinates, x="pos x", y="pos y",
                ax=ax, hue="fly_id", thresh=0.8, shade=True)

    plt.show()
