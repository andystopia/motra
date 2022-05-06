import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import figure, axes, colors
from matplotlib.animation import FuncAnimation, FFMpegWriter
from typing import Union

from .util import get_center_radius, sample_by_fly
from .constants import FPS


def _arena_boundary(arena_center: tuple[float],
                    arena_radius: float,
                    figsize: int = 10) -> tuple[figure.Figure, axes.Axes]:
    """ Draws a circle that represents the arena.

    Parameters
    ----------
    arena_center: tuple[float]
        Coordinates of the arena's center.

    arena_radius: float
        Radius of the arena.

    figsize: int. Optional, default to 10.
        Size of the figure.
        This argument will be passed as the figsize argument of matplotlib.pyplot.subplots.

    Returns
    -------
    matplotlib.figure.Figure, matplotlib.axes.Axes
    """

    fig, ax = plt.subplots(figsize=(figsize, figsize))

    arena_boundary = plt.Circle(
        arena_center, arena_radius, color="black", fill=False)
    ax.add_patch(arena_boundary)

    return fig, ax


def _remove_axes_labels(ax: plt.Axes) -> None:
    """ Removes the labels of the axes.
    """
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def arena_trajectory(coordinates: pd.DataFrame, arena_center: tuple,
                     arena_radius: int, figsize: int = 10) -> None:
    """ Draws the trajectory of each fly in the arena.

    Parameters
    ----------
    coordinates: pd.DataFrame
        Coordinates of each fly in the arena.

    arena_center: tuple[float]
        Coordinates of the arena's center.

    arena_radius: float
        Radius of the arena.

    figsize: int. Optional, default to 10.
        Size of the figure.
        This argument will be passed as the figsize argument of matplotlib.pyplot.subplots.

    Returns
    -------
    None
    """

    _, ax = _arena_boundary(arena_center, arena_radius, figsize)

    for fly_id in coordinates["fly_id"].unique():
        label = "fly {}".format(fly_id)
        ax.plot("pos x", "pos y",
                data=coordinates.loc[coordinates["fly_id"] == fly_id], label=label)
        _remove_axes_labels(ax)

    ax.legend()
    plt.show()


def heatmap(coordinates: pd.DataFrame, arena_center: tuple,
            arena_radius: int, figsize: int = 8, bins: int = 100,
            linthresh: float = 15, cmap: Union[colors.Colormap, str] = "viridis") -> None:
    """ Draws a heatmap that shows areas with most concentration of coordinates in the arena.

    Parameters
    ----------
    coordinates: pd.DataFrame
        Coordinates of each fly in the arena.

    arena_center: tuple[float]
        Coordinates of the arena's center.

    arena_radius: float
        Radius of the arena.

    figsize: int. Optional, default to 8.
        Size of the figure.
        This argument will be passed as the figsize argument of matplotlib.pyplot.subplots.

    bins: int. Optional, default to 100.
        Number of bins each dimension of the arena will be divided into.
        The larger number of bins is, the "higher-resolution" the heatmap is.

    linthresh: float. Optional, default to 15.
        Display threshold.

    cmap: matplotlib.colors.Colormap or str. Optional, default to 'viridis'.
        Color palette.
        This argument will be passed as the cmap argument of seaborn.heatmap.
        Options: https://matplotlib.org/stable/tutorials/colors/colormaps.html.

    Returns
    -------
    None
    """

    _, ax = _arena_boundary(arena_center, arena_radius, figsize)

    ax.hist2d(coordinates["pos x"], coordinates["pos y"],
              bins=bins, norm=colors.SymLogNorm(linthresh=linthresh), cmap=cmap)
    _remove_axes_labels(ax)

    plt.show()


def fly_animation(coordinates: pd.DataFrame, result_video_path: str,
                  video_size: float = None, figsize: int = 15) -> None:
    """ Creates animation of the flies in one arena over time.

    Parameters
    ----------
    coordinates: pd.DataFrame
        Coordinates of each fly in the arena.

    result_video_path: str
        Location to save the animation.

    video_size: float. Optional, default to None.
        In seconds.
        The function will only create animation of the first video_size seconds.
        If None, create animation of the whole video.

    figsize: int. Optional, default to 15.
        Size of the figure.
        This argument will be passed as the figsize argument of matplotlib.pyplot.subplots.

    Returns
    -------
    None
    """

    # Set up the background
    center, radius = get_center_radius(coordinates)
    fig, ax = _arena_boundary(center, radius, figsize)
    _remove_axes_labels(ax)

    # Prepare the data
    sample_coordinates = sample_by_fly(coordinates, video_size)
    fly_ids = sample_coordinates["fly_id"].unique()
    total_frames = sample_coordinates[sample_coordinates["fly_id"]
                                      == fly_ids[0]].shape[0]

    points = []
    lines = []
    lw = 2

    for _ in fly_ids:
        point, = ax.plot([], [], "o")
        line, = ax.plot([], [], lw=lw)

        points.append(point)
        lines.append(line)

    # Init animation
    def init():

        for idx in range(len(fly_ids)):
            points[idx].set_data([], [])
            lines[idx].set_data([], [])
            lines[idx].set_label("fly {}".format(fly_ids[idx]))

        return points + lines

    # Update frame
    def animate(current_frame):

        for fly_id_idx in range(len(fly_ids)):
            fly_coord = sample_coordinates[sample_coordinates["fly_id"]
                                           == fly_ids[fly_id_idx]]

            x = fly_coord["pos x"].to_numpy()[current_frame]
            y = fly_coord["pos y"].to_numpy()[current_frame]
            points[fly_id_idx].set_data([x], [y])

            x_cum = fly_coord["pos x"].to_numpy()[:current_frame]
            y_cum = fly_coord["pos y"].to_numpy()[:current_frame]
            lines[fly_id_idx].set_data([x_cum], [y_cum])

        return points + lines

    interval = FPS * 1000  # convert to ms
    animation = FuncAnimation(fig, animate, init_func=init, frames=total_frames,
                              interval=interval)
    animation.save(result_video_path, writer=FFMpegWriter(fps=FPS))

    plt.legend()
    plt.show()
