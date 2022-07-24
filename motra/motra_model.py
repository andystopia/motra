from __future__ import annotations

import os
from typing import Tuple, Optional, Any

import pandas as pd
import numpy as np
from .config import SetupConfig, DataframeKeys, MatplotlibConfig
from dataclasses import dataclass
import matplotlib.pyplot as plt

from .smallestenclosingcircle import make_circle
from .util import distance, get_center_radius


@dataclass(frozen=True)
class QuadrantModel:
    top_left: "MotraModel"
    top_right: "MotraModel"
    bottom_right: "MotraModel"
    bottom_left: "MotraModel"

    def as_list(self) -> list["QuadrantModel"]:
        return [self.top_left, self.top_right, self.bottom_right, self.bottom_left]


def attempt_get_figure_size_from(value):
    if type(value) is int:
        return value, value
    if type(value) is tuple:
        return value
    raise f"Cannot interpret figure size from: {value}"


def normalize_array(array):
    return (array - array.min()) / (array.max() - array.min())


@dataclass(frozen=True)
class Coordinate:
    x: float
    y: float

    @staticmethod
    def from_tuple(tup: tuple[float, float]) -> "Coordinate":
        """
        Create a coordinate from a tuple of values.
        Parameters
        ----------
        tup

        Returns
        -------

        """
        return Coordinate(tup[0], tup[1])


class Lazy:
    """
    A naive implementation of a lazily generated value, a cache of sorts.
    you can generate the value "on-the-fly" with by calling the value property
    getter, and that will generate, save, and return the data.

    After you call the .value property once, the future calls should be O(1) and it
    should never call the generation function again.
    """

    def __init__(self, generation_func, value=None):
        # note you cannot type annotate this class
        # unless you can use generic properties, and I'm not sure
        # if python can do that yet.
        self.__value = None
        assert generation_func is not None
        self.generation_func = generation_func

    @property
    def value(self):
        # not sure if this is cheaper in python
        # than a polymorphic call, but I don't
        # care enough to bench that right now.
        if not self.has_value:
            self.__value = self.generation_func()
        return self.__value

    @property
    def has_value(self):
        return self.__value is not None


class MotraModel:
    # create a default configuration such that
    # the radius of the arena is 19 millimeters
    # and the frames per second is equal to 30.
    DEFAULT_SETUP_CONFIG = SetupConfig(30.0, 19.0)
    DEFAULT_DATAFRAME_KEYS = DataframeKeys()

    def __init__(self, data: pd.DataFrame, setup_config: SetupConfig,
                 dataframe_keys: DataframeKeys, history: list = None):
        """
        Constructs a new Motra model.

        Note that unless you have had to write a parser by hand outside of
        this class, consider using a method like read from excel, and
        reformatting your data to match the conditions outlined
        in that method signature. This method is probably **not**
        what you want to do.

        Preconditions:
        Data input must be `fly_id`, `timestamp`, `pos x`, `pos y`,

        Parameters
        ----------
        data: the raw data input used to calculate fly movement
        config: the setup config for the experiment.
        """
        self.data = data
        self.setup_config = setup_config if setup_config is not None else MotraModel.DEFAULT_SETUP_CONFIG
        self.dataframe_keys = dataframe_keys if dataframe_keys is not None else MotraModel.DEFAULT_DATAFRAME_KEYS
        self.__velocity = Lazy(self.__calculate_velocity)
        self.__history = history if history is not None else []

    def time_distribution_by_quadrant(self, center: tuple[float, float] = (0, 0)) -> pd.DataFrame:
        from motra import time_distribution_by_quadrant
        return time_distribution_by_quadrant(self.data, center)

    def get_fly_identifiers(self) -> list[str]:
        """
        Gets the names of all the flies in the dataset
        Returns
        -------
        the names of all the flies in the dataset
        """
        return list(self.data[self.dataframe_keys.fly_id].unique())

    def calculate_smooth_velocity(self, smoothing_seconds: float = 1) -> pd.DataFrame:
        """Calculates a mean, smoothed velocity over time.

        Parameters
        ----------
        smoothing_seconds: the number of seconds to smooth over.

        Returns
        -------
        a velocity vector with the appropriate smoothing applied.
        column names depend on motra model config.
        """
        sample_interval_length = int(smoothing_seconds * self.setup_config.frames_per_second)

        start = 0
        seconds = []
        items = []
        while start < self.data.shape[0]:
            dat = self.data[start: min(self.data.shape[0], start + sample_interval_length)]
            x, y = dat[self.dataframe_keys.x_pos], dat[self.dataframe_keys.y_pos]
            velocity = np.mean(distance(x, y, x.shift(1), y.shift(1))) * sample_interval_length
            items.append(velocity)
            seconds.append(start / self.setup_config.frames_per_second)
            start += sample_interval_length
        return pd.DataFrame({
            self.dataframe_keys.timestamp: seconds,
            "velocity": items
        })

    def plot_smooth_velocity(self, smoothing=1, axis=None, size: tuple[int, int] | int = 14, apply_legend=False, x_label: Optional[bool | str] = None,
                      y_label: Optional[bool | str] = None,
                      title: Optional[bool] = None, legend_location: Optional[str] = None, y_units = None):
        """
        Plots the smoothed velocity over time
        Parameters
        ----------
        smoothing: the unit in seconds to plot the smoothed velocity over.
        axis: the axis to plot on, if none, one will be generated.
        size: the size of the axis to plot onto, if generated.
        apply_legend: whether or not to apply a legened

        Returns
        -------
        the axis plotted on.
        """
        if axis is None:
            fig, axis = plt.subplots(1, 1, figsize=attempt_get_figure_size_from(size))

        models = [(identifier, self.retain_only_fly_with_id(identifier)) for identifier in self.get_fly_identifiers()]

        for i, model in models:
            vel = model.calculate_smooth_velocity(smoothing)
            axis.plot(vel[model.dataframe_keys.timestamp], vel["velocity"],
                      label=f"velocity of fly: `{i}`" if apply_legend else None)
        axis.plot(vel[self.dataframe_keys.timestamp], vel["velocity"])


        if x_label is not False:
            axis.set_xlabel(x_label if x_label is not None else "Seconds")
        if y_label is not False:
            if y_units is not None:
                axis.set_ylabel(y_label if y_label is not None else f"distance ({y_units}) per {smoothing} second")
            else:
                axis.set_ylabel(y_label if y_label is not None else f"distance per {smoothing} second")
        if apply_legend is True:

            axis.legend(loc=legend_location if legend_location is not None else "upper right")
        if title is not False:
            axis.set_title(title if title is not None else "Graph of velocity of flies vs time")

        return axis

    @staticmethod
    def cat(models: list["MotraModel"]) -> Optional[MotraModel]:
        """
        Combines multiple motra models into a single motra model.
        Note that if your models are different in fields,
        only the first one's fields will be preserved, so
        keep in mind naming otherwise code *will* break
        down the line.
        Parameters
        ----------
        models: the models that you want to concatenate

        Returns
        -------
        a model with all the models concatenated.
        """

        assert len(models) > 0, "Error: You cannot concat zero models."

        combined_data = pd.concat(map(lambda mod: mod.data, models))
        root = models[0]
        return root.derive(data=combined_data, append_to_history=["concatenation!"])

    def get_total_distance_traveled(self) -> pd.DataFrame:
        """
        Calculates how far each fly travels.
        This will only be reasonable if you
        are looking at one quadrant, or overlayed quadrants,
        which have been scaled to the arena size.
        Returns
        -------
        an array containing how far the flies have travelled
        """
        ids = []
        distances = []
        models = [(identifier, self.retain_only_fly_with_id(identifier)) for identifier in self.get_fly_identifiers()]
        for id, model in models:
            ids.append(id)
            vel = model.velocity.to_numpy()
            vel = vel[~np.isnan(vel)]
            distances.append(vel.cumsum()[-1])
        return pd.DataFrame({
            self.dataframe_keys.fly_id: ids,
            "total distance traveled": distances,
        })

    def get_distances_traveled_during_interval(self, interval: float = 1):
        sample_interval_length = int(interval * self.setup_config.frames_per_second)

        start = 0
        seconds = []
        items = []
        while start < self.data.shape[0]:
            dat = self.data[start: min(self.data.shape[0], start + sample_interval_length)]
            x, y = dat[self.dataframe_keys.x_pos], dat[self.dataframe_keys.y_pos]
            dist = distance(x, y, x.shift(1), y.shift(1)).sum()
            items.append(dist)
            seconds.append(start / self.setup_config.frames_per_second)
            start += sample_interval_length
        return pd.DataFrame({
            self.dataframe_keys.timestamp: seconds,
            "distances travelled": items
        })

    def get_stopping_boolean_array(self, interval_smoothing: float, stopping_epsilon: float):
        return self.get_distances_traveled_during_interval(interval_smoothing)[
                   "distances travelled"].to_numpy() <= stopping_epsilon

    def get_number_of_stops(self, interval_smoothing: float, stopping_epsilon: float) -> int:
        """
        Counts the number of times which a fly stops.
        Only available if you have one fly.
        Remember to scale down.
        Parameters
        ----------
        interval_smoothing: the amount of time to calculate the distance over
        stopping_epsilon: how slow is slow enough? Should be in units of mm / seconds.

        Returns
        -------
        the number of times that this fly stopped.
        """
        return self.get_stopping_boolean_array(interval_smoothing, stopping_epsilon)

    def get_time_of_sedation(self, interval_smoothing: float, stopping_epsilon: float):
        """
        Get the time that this fly was sedated.
        Only works if you have a singular fly.
        Parameters
        ----------
        interval_smoothing
        stopping_epsilon

        Returns
        -------

        """
        stops: np.ndarray = (self.get_distances_traveled_during_interval()[
                                 "distances travelled"].to_numpy() <= stopping_epsilon)[::-1]
        if np.all(stops == False):
            return "fly unsedated"
        try:
            print(stops.shape)
            first_false = len(stops) - np.where(stops == False)[0][0]
        except KeyError:
            return "fly unsedated"

        print(first_false)
        return self.data[
            self.dataframe_keys.timestamp].to_numpy()[first_false]

    def plot_distance_from_center_over_time(self, axis=None, size: tuple[int, int] | int = 14, xlab=True, ylab=True,
                                            title=True, apply_legend=False):
        """
        Plot a histogram of the distances from the center of the arena.
        Parameters
        ----------
        axis: the axis to plot on, can if none, then one will be created for plotting.
        size: if necessary, the size of the created axis to plot on.

        Returns
        -------
        the axis plotted on.
        """
        if axis is None:
            fig, axis = plt.subplots(1, 1, figsize=attempt_get_figure_size_from(size))

        for (id, fly) in self.get_flies():
            radius = np.sqrt(
                np.square(fly.data[fly.dataframe_keys.x_pos]) +
                np.square(fly.data[fly.dataframe_keys.y_pos])
            )
            axis.plot(fly.data[fly.dataframe_keys.timestamp], radius, label=f"fly w/ id: `{id}`", linewidth=3.0)

        if xlab:
            axis.set_xlabel(xlab if xlab is not True else "time (s)")
        if ylab:
            axis.set_ylabel(ylab if ylab is not True else "distance from center (mm)")
        if title:
            axis.set_title(title if title is not True else "distance from center (mm) vs time (s)")
        if apply_legend:
            axis.legend(loc="upper left" if apply_legend is True else apply_legend, fancybox=True,
                        bbox_to_anchor=(1, 0.95))
        return axis

    def plot_distance_from_center_histogram(self, axis=None, size: tuple[int, int] | int = 14):
        """
        Plot a histogram of the distances from the center of the arena.
        Parameters
        ----------
        axis: the axis to plot on, can if none, then one will be created for plotting.
        size: if necessary, the size of the created axis to plot on.

        Returns
        -------
        the axis plotted on.
        """
        if axis is None:
            fig, axis = plt.subplots(1, 1, figsize=attempt_get_figure_size_from(size))

        radius = np.sqrt(
            np.square(self.data[self.dataframe_keys.x_pos]) +
            np.square(self.data[self.dataframe_keys.y_pos])
        )

        axis.hist(radius)
        return axis

    def get_flies(self) -> list[tuple[Any, "MotraModel"]]:
        """
        Returns a list of fly id's with the respective fly model.
        Returns
        -------

        """
        return [(identifier, self.retain_only_fly_with_id(identifier)) for identifier in self.get_fly_identifiers()]

    def plot_velocity(self, axis=None, size: tuple[int, int] | int = 14,
                      apply_legend: bool = False, x_label: Optional[bool | str] = None,
                      y_label: Optional[bool | str] = None,
                      title: Optional[bool] = None, legend_location: Optional[str] = None):
        """
        Plots the velocity of the flies against time.
        It's recommended that you set the scale of the arena
        before completing this operation so that all the
        velocities are per mm.
        Parameters
        ----------
        axis: the axis to plot on, if none will automatically create a new axis
        smoothing: the number of seconds to smooth the graph over by.
        size: the size of the axis to plot on. By default this is 14, unused if axis is specified.
        apply_legend: set to true, if you would like a legend.
        x_label: if none, then will generate an xlabel automatically, if false, will not genreate an x label, if
        passed a string, will use that string as the x label.
        y_label: if none, will generate a y lable automatically, if false, will leave the y label blank, if
        passed a stirng, will use that passed string as the y label.
        title: if none, will generate a title, if false, will leave the title blank, if passed a string, will
        use that stirng as the title.
        legend_location: if none, will place the legend, in the upper right, else, if passed a string,
        will pass that string through to the matplotlib legend function. for reference on
        valid strings to pass, view matplotlib's documentation on legends.

        Returns
        -------
        the axis that the velocity was plotted on.
        """

        ## we will implement a smoothing algorithm, by simply
        if axis is None:
            fig, axis = plt.subplots(1, 1, figsize=attempt_get_figure_size_from(size))

        models = [(identifier, self.retain_only_fly_with_id(identifier)) for identifier in self.get_fly_identifiers()]

        for i, model in models:
            axis.plot(model.data[model.dataframe_keys.timestamp], model.velocity,
                      label=f"velocity of fly: `{i}`" if apply_legend else None)

        if x_label is not False:
            axis.set_xlabel(x_label if x_label is not None else "Seconds")
        if y_label is not False:
            axis.set_ylabel(y_label if y_label is not None else "distance per second")
        if apply_legend is True:
            axis.legend(loc=legend_location if legend_location is not None else "upper right")
        if title is not False:
            axis.set_title(title if title is not None else "Graph of velocity of flies vs time")
        return axis

    def retain_only_fly_with_id(self, identifier):
        """
        Select only the fly with the given identifier.

        Parameters
        ----------
        identifier the identifier of the fly to retain.

        Returns
        -------
        a new motra model containing only the given fly.
        """
        return self.derive(data=self.data[self.data[self.dataframe_keys.fly_id] == identifier],
                           append_to_history=f"filtered out all flies whose id was not: `{identifier}`")

    def get_average_distance_over_time_interval(self, time_interval: float):
        """
        Calculates the average distance which a fly travels, on average,
        throughout a given time interval.

        Parameters
        ----------
        time_interval the amount of time to sample from.
        Returns
        -------
        a dataframe summarizing how far the fly traveled over varoius
        time intervals.
        """
        coordinates = self.data.copy()
        # Calculates total distance each fly travels within a time interval.
        coordinates["distance"] = coordinates.groupby("fly_id")[
            "distance"].cumsum()

        coordinates["index_in_group"] = coordinates.groupby("fly_id").cumcount()
        frames = time_interval * self.setup_config.frames_per_second

        coordinates["mod_of_index"] = coordinates["index_in_group"] % frames
        coordinates = coordinates.loc[coordinates["mod_of_index"] == 0]
        coordinates.loc[coordinates["index_in_group"] == 0, "distance"] = 0

        coordinates["distance"] = coordinates["distance"] - \
                                  coordinates.groupby("fly_id")["distance"].shift(1)

        # Retrieves columns of interest and calculates velocity
        result = coordinates[["fly_id", "timestamp", "distance"]].copy()
        result["velocity (per second)"] = result["distance"] / time_interval
        return result

    def extract_time_range(self, starting_second: float | None, ending_second: float | None) -> "MotraModel":
        """
        Extracts an explicit time range from the video sequence from
        starting time to ending time. Note that if you want to slice
        *from* the start to some value, use `extract_time_range(None, value)`
        and to extract *from* some value to the end, use
        `extract_time_range(value, None)`. And to extract between two
        different seconds, (a, b) in the video, use, `extract_time_range(a, b)`.

        Keep in mind because the video is sampled more than once per second,
        it may be helpful to pass a decimal value in, as in
        `extract_time_range(1.5, None)`, so keep that in mind.

        Instead of passing None, as well, you can also pass ±float('inf'),
        depending on the starting or ending second, and that will also
        extract correctly. This is mostly only helpful when used
        in a single-typed environment for simplicity, but has no real
        advantages over passing None.

        Also note that this follows standard of having the start be "closed"
        and the end be "open". To put it another way, the interval of rationals
        which are passed to this function is "right-open", of the form [a, b).

        Parameters
        ----------
        starting_second the second of the video to start at.
        ending_second the second of the video to end at

        Returns
        -------
        a new motra model containing only the time range requested.
        """

        start = starting_second if starting_second is not None else -float('inf')
        end = starting_second if ending_second is not None else float('inf')
        indicies = np.logical_and(self.data[self.dataframe_keys.timestamp] > start,
                                  self.data[self.dataframe_keys] <= end)
        if starting_second is None and ending_second is None:
            history_message = "Extracted fly paths over the entire time range"
        elif starting_second is None:
            history_message = f"Extracted fly paths up to second {end}"
        elif ending_second is None:
            history_message = f"Extracted fly paths from the {start} second to the end of recording"
        else:
            history_message = f"Extracted fly paths from time interval (in seconds): [{start}, {end}]"
        return MotraModel(self.data[indicies], self.setup_config, self.dataframe_keys,
                          self.__history + [history_message])

    @property
    def history(self) -> str:
        """
        Get the history of this model.
        Returns
        -------

        """
        history_value = ""
        for counter, value in enumerate(self.__history):
            history_value += f"{counter + 1}. {value}\n"
        return history_value

    @property
    def velocity(self):
        """
        Calculates the velocity of the fly.
        This method only can logically be
        called, if you have exactly one fly
        selected. Calling this on multiple flies
        will lead to extraneous, silent mistakes
        at the endpoints.
        Returns
        -------
        the velocity of the flies over time.
        """
        return self.__velocity.value

    def scale_to_arena_size(self):
        """
        Scales the data up to the defined arena size. Remember you can configure this
        parameter only when you parse the csv file, so be sure to do it then.

        Returns
        -------
        a new dataframe with the scaling applied.
        """
        data_center_x, data_center_y, data_radius = make_circle(self.get_positions(True).to_numpy())
        # center our data at the center of the arena.
        locations = self.get_positions() - np.array([data_center_x, data_center_y])

        # convert to polar coordinates
        theta = np.arctan2(locations[self.dataframe_keys.y_pos], locations[self.dataframe_keys.x_pos])
        radius = np.sqrt(
            np.square(locations[self.dataframe_keys.x_pos]) +
            np.square(locations[self.dataframe_keys.y_pos])
        )

        # normalize and upscale the radius
        radius = (radius / data_radius) * self.setup_config.arena_radius

        # convert back to cartesian
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        # create a new dataframe
        data = self.data.copy()
        data[self.dataframe_keys.x_pos] = x
        data[self.dataframe_keys.y_pos] = y

        return self.derive(data=data, append_to_history=f"Scaled to arena radius: {self.setup_config.arena_radius}mm")

    def __calculate_velocity(self) -> np.ndarray:
        """
        Calculates the velocity of the flies.
        Returns
        -------
        an array containing the velocities of the flies.
        """
        return distance(self.get_x_pos(), self.get_y_pos(), self.get_x_pos().shift(1), self.get_y_pos().shift(1))

    def get_x_pos(self, copy=False):
        """
        Get the x position variable in this dataframe
        Parameters
        ----------
        copy: makes a copy if set to true, otherwise, returns the underlying array.

        Returns
        -------
        the x positions of the flies
        """
        return self.data[self.dataframe_keys.x_pos]

    def get_y_pos(self, copy=False):
        """
        Get the y position dataframe in this variable
        Parameters
        ----------
        copy: makes a copy if set to true, otherwise, returns the underlying array.

        Returns
        -------
        the y positions of the flies
        """
        return self.data[self.dataframe_keys.y_pos]

    def get_positions(self, copy=False):
        """
        Gets the combined x y positions as a pandas dataframe
        Parameters
        ----------
        copy: makes a copy if set to true else does not make a copy.

        Returns
        -------
        a dataframe of the positions of the flies
        """
        temp = self.data[[self.dataframe_keys.x_pos, self.dataframe_keys.y_pos]]
        temp = temp if copy is False else temp.copy()
        return temp

    def filter_quadrant(self, arena_center_coordinate: Coordinate | tuple[float, float],
                        center_data_at_arena_center: bool = False) -> "MotraModel":
        """
        Filters out a MotraModel down to a single quadrant worth of data.
        Use this when you want to be able to include and exclude various
        sections of the arena.
        Parameters
        ----------
        arena_center_coordinate the center most coordinate of the arena
        center_data_at_arena_center determines whether the data be centered at the arena center.

        Returns
        -------
        a new motra model with only the data from a single quadrant.
        """
        # Note that the following code has a certain logical assumption
        # notably that if a fly has a coordinate within a certain radius
        # of an arena center, it must be in that arena, which I think is
        # fine, because flies should not be "jumping" from one arena
        # to the next.

        # convert tuples to coordinates, because
        # we want to allow passing tuples in as arguments to reduce typing
        # and make it easier for library users, but still want the convenience and
        # safety of a full-blown object.
        if type(arena_center_coordinate) is tuple:
            arena_center_coordinate = Coordinate.from_tuple(arena_center_coordinate)

        assert type(arena_center_coordinate) is Coordinate
        # shift all points such that they are centered at the arena center coordinate,
        # any points within the given radius of that origin, must be within that circle.

        # small optimization to not square root the radius, because
        # we don't want to lose fp precision, and it will save a few cycles
        # here and there.
        squared_distances_from_center = (self.data[self.dataframe_keys.x_pos] - arena_center_coordinate.x) ** 2 + \
                                        (self.data[self.dataframe_keys.y_pos] - arena_center_coordinate.y) ** 2

        # square the arena radius
        arena_squared_radius = self.setup_config.arena_radius ** 2

        # save only the valid coordinates, note the fact that this "boolean comparison" generates a
        # series, because the operation is completed on a series
        valid_coordinates: pd.Series[bool] = squared_distances_from_center <= arena_squared_radius

        # only save the valid coordinates in the dataset
        filtered = self.data[valid_coordinates].copy()

        history_message = f"Filtered by Quadrant {arena_center_coordinate}"
        if center_data_at_arena_center:
            filtered[self.dataframe_keys.x_pos] -= arena_center_coordinate.x
            filtered[self.dataframe_keys.y_pos] -= arena_center_coordinate.y
            history_message += ", centered fly paths at the arena center"

        # return a new motra model which only has the filtered flies.
        return self.derive(data=filtered, append_to_history=history_message)

    def normalize(self):
        """
        Normalize the data set to a range of zero to one.

        If you have centered the dataset at the middle of
        an arena, this could maek sense to do for various reasons.

        I guess if you want to, you could normalize the entire raw data set, and
        if you really want to work in [0, 1] coordinates all the time.
        Returns
        -------
        a normalized Motra Model where all coordinates are between zero and one.
        """
        copied = self.data.copy()

        # consider using 2D vector normalization here.
        copied[self.dataframe_keys.x_pos] = normalize_array(copied[self.dataframe_keys.x_pos])
        copied[self.dataframe_keys.y_pos] = normalize_array(copied[self.dataframe_keys.y_pos])

        return self.derive(data=copied, append_to_history="Normalized Motra Model to Range: [0, 1]")

    def overlay_quadrants(self,
                          splitting_origin_point: Coordinate | tuple,
                          *,  # keep them on their toes!
                          include_top_left=True,
                          include_top_right=True,
                          include_bottom_left=True,
                          include_bottom_right=True) -> "MotraModel":
        """
        Splits the quadrants up and then centers them, scales them to the
        arena size.
        Parameters
        ----------
        splitting_origin_point: the point at which to split the dataset at.
        include_top_left: whether to include the top left quadrant
        include_top_right: whether to include the top right quadrant
        include_bottom_left: whether to include the bottom left quadrant
        include_bottom_right: whether to include the bottom right quadrant

        Returns
        -------
        a new motra model with the quadrants overlaid.
        """
        quadrants: QuadrantModel = self.split_up_quadrants(splitting_origin_point)

        # create an array of the quadrants and then
        # index them using the booleans to specify which ones we want.
        quadrant_array = np.array([
            quadrants.top_left,
            quadrants.top_right,
            quadrants.bottom_left,
            quadrants.bottom_right])[
            np.array([include_top_left, include_top_right, include_bottom_left, include_bottom_right])]
        print(quadrant_array)
        return MotraModel.cat([quadrant.scale_to_arena_size() for quadrant in quadrant_array])

    def split_up_quadrants(self, splitting_origin_point: Coordinate | tuple,
                           auto_center_fly_data: bool = False) -> QuadrantModel:
        """
        Divides up the data into the different quadrant sections of
        the arena setup.

        By passing a splitting_origin_point, if we allow that point to be the new, translated origin of
        the 2D plane, if the fly data is more "upper left" of that point, then it will be assigned
        to the upper left quadrant in the resultingly returned QuadrantModel.


        If the lines of the arena container are parallel to the respective frame borders of the camera, and the
        center of the arena is in the center of the camera, then you can pass the center pixel of the frame
        in for the coordinate and it will split up each quadrant of the arena, as is most expected.

        Parameters
        ----------
        splitting_origin_point
        the point which splits up the quadrants.
        auto_center_fly_data
        if set to true, derive a circle which bounds the flies in each quadrant of the arena, and then
        center the flies coordinates at the middle of that derived circle. This allows you to relate the
        flies to motion relative to their arena that they sit in, not relative to your camera pixel-space
        setup. Generally, for analytical methods, you probably want to set this to be true, but for intermediate
        steps you probably want this parameter to be false.
        Returns
        -------
        a quadrant model object which will holds all the quadrants of the entire physical fly container model.
        """
        # implicity attempt to convert the splitting point to a coordinate
        if type(splitting_origin_point) is tuple:
            splitting_origin_point = Coordinate.from_tuple(splitting_origin_point)

        assert type(splitting_origin_point) is Coordinate

        upper_indicies = self.data[self.dataframe_keys.y_pos] > splitting_origin_point.y
        lower_indicies = self.data[self.dataframe_keys.y_pos] < splitting_origin_point.y
        right_indicies = self.data[self.dataframe_keys.x_pos] > splitting_origin_point.x
        left_indicies = self.data[self.dataframe_keys.x_pos] < splitting_origin_point.x

        upper_right_dataframe = self.data[np.logical_and(upper_indicies, right_indicies)]
        upper_left_dataframe = self.data[np.logical_and(upper_indicies, left_indicies)]
        lower_right_dataframe = self.data[np.logical_and(lower_indicies, right_indicies)]
        lower_left_dataframe = self.data[np.logical_and(lower_indicies, left_indicies)]

        if auto_center_fly_data:
            upper_right_dataframe[[self.dataframe_keys.x_pos, self.dataframe_keys.y_pos]] -= np.array(
                get_center_radius(upper_right_dataframe)[0])
            upper_left_dataframe[[self.dataframe_keys.x_pos, self.dataframe_keys.y_pos]] -= np.array(
                get_center_radius(upper_left_dataframe)[0])
            lower_right_dataframe[[self.dataframe_keys.x_pos, self.dataframe_keys.y_pos]] -= np.array(
                get_center_radius(lower_right_dataframe)[0])
            lower_left_dataframe[[self.dataframe_keys.x_pos, self.dataframe_keys.y_pos]] -= np.array(
                get_center_radius(lower_left_dataframe)[0])

        return QuadrantModel(
            top_left=self.derive(data=upper_left_dataframe,
                                 append_to_history="Extracted upper left quadrant of fly model setup"),
            top_right=self.derive(data=upper_right_dataframe,
                                  append_to_history="Extracted upper right quadrant of fly model setup"),
            bottom_left=self.derive(data=lower_left_dataframe,
                                    append_to_history="Extracted lower left quadrant of fly model setup"),
            bottom_right=self.derive(data=lower_right_dataframe,
                                     append_to_history="Extracted lower right quadrant of fly model setup"),
        )

    def plot(self, axis: plt.Axes = None, size: int | tuple[int, int] = 14) -> plt.Axes:
        """
        plots the paths of all flies over all time in the model.

        if you pass an axis it will graph on that axis, otherwise, it will
        graph on a default generated axis.
        Parameters
        ----------
        axis: the axis to graph on, may be left as none, and will generate an axis to graph on if so.
        size: the size of the created axis to graph on, if an integer, will generate a square plot, or
        you can define an arbritary scaling by passing a tuple.

        Returns
        -------
        the axis that the function was graphed on.
        """
        if axis is None:
            fig, axis = plt.subplots(1, 1, figsize=attempt_get_figure_size_from(size))
        axis.scatter(self.data[self.dataframe_keys.x_pos], self.data[self.dataframe_keys.y_pos],
                     c=self.data[self.dataframe_keys.fly_id])

        return axis

    def plot_flies_at_frame(self, frame: int, axis: plt.Axes = None, figure_size=None, auto_title=False) -> plt.Axes:
        """Plot the locations of flies at a variety of locations.

        Parameters
        ----------
        frame: the frame to extract
        axis : the axis to plot the locations on. leaving as none is completely fine, it will just create an axis and
        plot on it.
        figure_size : the figure size to plot at, only works if you pass none as the axis parameter
        auto_title : boolean, set to true if you just want an autogenerated plot title.

        Returns
        -------
        the axis that this plot was drawn on.
        """
        if axis is None:
            fig, axis = plt.subplots(1, 1, figsize=figure_size)

        second = frame / self.setup_config.fps

        if auto_title:
            axis.set_title(f"Graph of the flies at frame {frame} aka {second} second")

        # check whether within frame epsilon
        boolean_selector = np.abs(self.data[self.dataframe_keys.timestamp].to_numpy() - second) < 1e-6
        graph_data: pd.DataFrame = self.data[boolean_selector]
        x_data = graph_data[self.dataframe_keys.x_pos].to_numpy()
        y_data = graph_data[self.dataframe_keys.y_pos].to_numpy()

        axis.scatter(x_data, y_data, c=graph_data[self.dataframe_keys.fly_id])
        return axis

    def derive(self,
               *,
               data: pd.DataFrame | None = None,
               setup_config: SetupConfig | None = None,
               dataframe_keys: DataframeKeys | None = None,
               history: list | None = None,
               append_to_history: str | None
               ) -> "MotraModel":
        """
        Creates a "clone"
        derive a new motra model from an existing one.
        this method is useful if you have a dataframe, and
        then, say you do a non-mutating modification, and you
        want to generate a new dataframe.

        All properties are, by default, None, and if you leave
        them as None, then you will simply be using the data that
        already sits in this instance.
        Parameters
        ----------
        data: the data to use
        setup_config: the setup config to use
        dataframe_keys: the dataframe keys to use
        history: the history to use
        append_to_history: the message that should be appended to the history after the clone

        Returns
        -------
        a new motra model which has the passed properties or the properties of the current instance.
        """
        history = self.__history if history is None else history
        if append_to_history is not None:
            processed_history = history + [append_to_history]
        else:
            processed_history = history
        return MotraModel(
            data if data is not None else self.data,
            setup_config if setup_config is not None else self.setup_config,
            dataframe_keys if dataframe_keys is not None else self.dataframe_keys,
            processed_history if processed_history is not None else self.__history
        )

    @staticmethod
    def read_csv(path: os.PathLike, setup_config: Optional[SetupConfig] = None,
                 dataframe_key_config: Optional[DataframeKeys] = None) -> "MotraModel":
        """
        Reads in a fly tracking file from a csv file and constructs
        a new MotraModel from the data.

        Parameters
        ----------
        path: the path to the fly tracking data.
        setup_config: the setup config to use, if left as none, will use the default.
        dataframe_key_config: the dataframe keys to use for the generated dataframe, if none, will use default

        Returns
        -------
        a motra model containing the information at the given path
        """
        # establish default configs.
        setup_config = setup_config if setup_config is not None else MotraModel.DEFAULT_SETUP_CONFIG
        dataframe_key_config = dataframe_key_config if dataframe_key_config is not None \
            else MotraModel.DEFAULT_DATAFRAME_KEYS

        ###
        # this method generates a dataframe with four columns such that.
        # `fly_id`, `timestamp`, `pos x`, `pos y`,
        # are the four columns.
        ###

        df = pd.read_csv(path)
        del df["frame"]
        # del df[" "]

        x_cols = [col for col in df.columns if "x" in col]
        y_cols = [col for col in df.columns if "y" in col]

        combined_data = df[x_cols].melt()
        y_vals = df[y_cols].melt()["value"]

        combined_data.rename(columns={"value": dataframe_key_config.x_pos}, inplace=True)
        combined_data[dataframe_key_config.y_pos] = y_vals
        combined_data.rename(columns={"variable": dataframe_key_config.fly_id}, inplace=True)

        fly_ids = combined_data[dataframe_key_config.fly_id].unique()

        time_stamps = np.tile(np.arange(df.shape[0]) / setup_config.fps, len(fly_ids))
        combined_data[dataframe_key_config.timestamp] = time_stamps

        fly_idx = np.arange(1, len(fly_ids) + 1)

        dict_lookup = dict(zip(fly_ids, fly_idx))

        combined_data[dataframe_key_config.fly_id].replace(to_replace=dict_lookup, inplace=True)

        return MotraModel(combined_data, setup_config, dataframe_key_config,
                          history=[f"read from blender generated csv at path: {path}"])

    @staticmethod
    def read_from_excel(path: os.PathLike, setup_config: Optional[SetupConfig] = None,
                        dataframe_key_config: Optional[DataframeKeys] = None) -> "MotraModel":
        """Parses coordinates data from an Excel file.

        Creates a pandas dataframe which has four
        columns,
        `fly_id`, `timestamp`, `pos x`, `pos y`.
        The input Excel file must have a separate tab for each fly.
        Each tab must have 2 columns: 'pos x' and 'pos y'.

        Parameters
        ----------
        path: str
            Path to the Excel file of coordinates.

        setup_config: the configuration the project was run under,
            basic things like camera fps and arena radius. If None
            then it will use the

        dataframe_key_config: the key names of the dataframe to generate,
            however, if None, then it will use the default config, specified in this class.

        Returns
        -------
        pandas.DataFrame
            Coordinates of every fly in the experiment over time.
        """

        # establish default configs.
        setup_config = setup_config if setup_config is not None else MotraModel.DEFAULT_SETUP_CONFIG
        dataframe_key_config = dataframe_key_config if dataframe_key_config is not None \
            else MotraModel.DEFAULT_DATAFRAME_KEYS

        ###
        # this method generates a dataframe with four columns such that.
        # `fly_id`, `timestamp`, `pos x`, `pos y`,
        # are the four columns.
        ###
        flies_dict = pd.read_excel(path, sheet_name=None)
        decimal_precision = setup_config.decimal_precision

        count = 1
        for fly in flies_dict.values():
            if not fly.empty:
                # Assign id for each fly
                fly[dataframe_key_config.fly_id] = count

                # Generate timestamp based on camera's FPS.
                fly[dataframe_key_config.timestamp] = np.round(
                    np.arange(fly.shape[0]) / setup_config.fps, decimal_precision)

                count += 1

        flies = pd.concat(flies_dict.values(), ignore_index=True)
        return MotraModel(flies, setup_config, dataframe_key_config, history=[f"read from excel at path: `{path}`"])
