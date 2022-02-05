import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import seaborn as sns


def arena(coordinates: pd.DataFrame) -> tuple[tuple, int]:

    center = (0, 0)
    radius = 0

    coordinates_copy = coordinates.copy()
    coordinates_copy.reset_index(drop=True, inplace=True)

    min_x_idx = coordinates_copy["pos x"].idxmin()
    max_x_idx = coordinates_copy["pos x"].idxmax()
    min_y_idx = coordinates_copy["pos y"].idxmin()
    max_y_idx = coordinates_copy["pos y"].idxmax()

    diameter_x = coordinates_copy.iloc[max_x_idx]["pos x"] - \
        coordinates_copy.iloc[min_x_idx]["pos x"]
    diameter_y = coordinates_copy.iloc[max_y_idx]["pos y"] - \
        coordinates_copy.iloc[min_y_idx]["pos y"]

    if diameter_x < diameter_y:
        center_x = (coordinates_copy.iloc[min_y_idx]["pos x"] +
                    coordinates_copy.iloc[max_y_idx]["pos x"]) / 2
        center_y = (coordinates_copy.iloc[min_y_idx]["pos y"] +
                    coordinates_copy.iloc[max_y_idx]["pos y"]) / 2
        radius = diameter_y / 2
    else:
        center_x = (coordinates_copy.iloc[min_x_idx]["pos x"] +
                    coordinates_copy.iloc[max_x_idx]["pos x"]) / 2
        center_y = (coordinates_copy.iloc[min_x_idx]["pos y"] +
                    coordinates_copy.iloc[max_x_idx]["pos y"]) / 2
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
            arena_radius: int, figsize: int = 10, thresh: float = 0.5) -> None:

    ax = _arena_boundary(arena_center, arena_radius, figsize)

    sns.kdeplot(data=coordinates, x="pos x", y="pos y",
                ax=ax, hue="fly_id", thresh=thresh, fill=True, palette="tab10")

    plt.show()


def fly_animation(coordinates: pd.DataFrame, result_video_path: str):
    x_coords = coordinates["pos x"]
    y_coords = coordinates["pos y"]

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_xlim(min(x_coords), max(x_coords))
    ax.set_ylim(min(y_coords), max(y_coords))
    point, = ax.plot([], [], "bo")
    line, = ax.plot([], [], lw=2)

    def init():
        point.set_data([], [])
        line.set_data([], [])
        return line, point

    def ani(coords):
        point.set_data([coords[1]], [coords[2]])
        line.set_data(x_coords[:coords[0]], y_coords[:coords[0]])
        return line, point

    def frames():
        index = 0
        for acc_11_pos, acc_12_pos in zip(x_coords, y_coords):
            yield index, acc_11_pos, acc_12_pos
            index += 1

    anim = FuncAnimation(fig, ani, init_func=init,
                         frames=frames, interval=1000/30, save_count=coordinates.shape[0])
    anim.save(result_video_path, writer=FFMpegWriter(fps=30))

    plt.show()
