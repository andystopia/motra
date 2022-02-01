from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _arena(arena_center: tuple, arena_radius: int, figsize: int = 10):

    _, ax = plt.subplots(figsize=(figsize, figsize))

    arena_boundary = plt.Circle(
        arena_center, arena_radius, color="black", fill=False)
    ax.add_patch(arena_boundary)

    return ax


def arena_trajectory(coordinates: pd.DataFrame, arena_center: tuple,
                     arena_radius: int, figsize: int = 10) -> None:

    ax = _arena(arena_center, arena_radius, figsize)

    for fly_id in coordinates["fly_id"].unique():
        label = "fly {}".format(fly_id)
        ax.plot("pos x", "pos y",
                data=coordinates.loc[coordinates["fly_id"] == fly_id], label=label)

    ax.legend()
    plt.show()


def heatmap(coordinates: pd.DataFrame, arena_center: tuple,
            arena_radius: int, figsize: int = 10) -> None:

    ax = _arena(arena_center, arena_radius, figsize)

    sns.kdeplot(data=coordinates, x="pos x", y="pos y",
                ax=ax, hue="fly_id", thresh=0.8, shade=True)

    plt.show()
