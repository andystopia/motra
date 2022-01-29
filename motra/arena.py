import pandas as pd


def arena(coordinates: pd.DataFrame) -> tuple[tuple, int]:

    center = (0, 0)
    radius = 0

    min_x_idx = coordinates["pos x"].idxmin()
    max_x_idx = coordinates["pos x"].idxmax()
    min_y_idx = coordinates["pos y"].idxmin()
    max_y_idx = coordinates["pos y"].idxmax()

    radius_x = coordinates.iloc[max_x_idx]["pos x"] - \
        coordinates.iloc[min_x_idx]["pos x"]
    radius_y = coordinates.iloc[max_y_idx]["pos y"] - \
        coordinates.iloc[min_y_idx]["pos y"]

    if radius_x < radius_y:
        center_x = (coordinates.iloc[min_y_idx]["pos x"] +
                    coordinates.iloc[max_y_idx]["pos x"]) / 2
        center_y = (coordinates.iloc[min_y_idx]["pos y"] +
                    coordinates.iloc[max_y_idx]["pos y"]) / 2
        radius = radius_y
    else:
        center_x = (coordinates.iloc[min_x_idx]["pos x"] +
                    coordinates.iloc[max_x_idx]["pos x"]) / 2
        center_y = (coordinates.iloc[min_x_idx]["pos y"] +
                    coordinates.iloc[max_x_idx]["pos y"]) / 2
        radius = radius_x

    center = (center_x, center_y)

    return center, radius
