from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt


@dataclass
class MatplotlibConfig:
    # set the ggplot style to be the default matplotlib
    # schema
    style_name: str = 'ggplot'
    font_size: int = 22

    def apply(self):
        plt.style.use(self.style_name)
        plt.rcParams.update({'font.size': self.font_size})

@dataclass(frozen=True)
class SetupConfig:
    frames_per_second: float
    arena_radius_in_mm: float

    # establish a reasonable default
    # for the decimal precision. AFAIK,
    # there is nothing "special" about
    # this "special" number, change to whatever
    # makes you happy. :)
    decimal_precision: int = 4

    @property
    def fps(self):
        """
        Short Alias for Frames Per Second value
        Returns
        -------

        """
        return self.frames_per_second

    @property
    def arena_radius(self):
        """
        Short Alias for the radius of reach well in
        millimeters
        Returns
        -------
        """
        return self.arena_radius_in_mm



@dataclass
class DataframeKeys:
    """
    Contains the various keys and ways of accessing
    the main dataframe which holds all the fly data.

    I, as a general rule, do not use strings to
    express logical selection, however, python really
    likes that error-prone style, so instead, I throw
    them all in here, that way my IDE warns me if
    I make a mistake.
    """
    x_pos: str = "pos x"
    y_pos: str = "pos y"
    timestamp: str = "timestamp"
    fly_id: str = "fly_id"
    fly_id_generation_func: Callable[[int], str] = lambda val: f"fly {val}"

    def get_fly_key_by_id(self, fly_id: int):
        return self.fly_id_generation_func(fly_id)
