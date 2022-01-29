import pandas as pd


def stats(flies_data: pd.DataFrame, frames: int = 30) -> pd.DataFrame:
    """
    Generates statistics (distance, velocity, etc.) based on number of frames.
    Parameters
    ----------
    coordinates: pd.DataFrame

    frames: int
        number of frames
    """

    # flies_data.groupby(flies_data.index / frames).apply(lambda )

    coordinates = flies_data[["pos x", "pos y"]]
    # coordinates[["pos_x_lagged", "pos_y_lagged"]
    # ] = coordinates.groupby(flies_data.index / frames)["pos x"].shift(1)

    return coordinates.groupby(coordinates.index / frames)["pos x"].mean()
