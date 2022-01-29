import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def arena_trajectory(coordinates: pd.DataFrame, arena_center: tuple,
                     arena_radius: int, figsize: int = 10) -> None:

    _, ax = plt.subplots(figsize=(figsize, figsize))

    arena_boundary = plt.Circle(
        arena_center, arena_radius, color="black", fill=False)
    ax.add_patch(arena_boundary)

    sns.lineplot(data=coordinates, x="pos x", y="pos y", hue="fly_id", ax=ax)

    return ax
