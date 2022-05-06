import pandas as pd
import numpy as np
from .constants import FPS


def parse(path: str) -> pd.DataFrame:
    """Parses coordinates data from an Excel file.
    The input Excel file must have a separate tab for each fly. 
    Each tab must have 2 columns: 'pos x' and 'pos y'.

    Parameters
    ----------
    path: str
        Path to the Excel file of coordinates.

    Returns
    -------
    pandas.DataFrame
        Coordinates of every fly in the experiment over time.
    """

    flies_dict = pd.read_excel(path, sheet_name=None)
    decimal_precision = 4

    count = 1
    for fly in flies_dict.values():
        if not fly.empty:

            # Assign id for each fly
            fly["fly_id"] = count

            # Generate timestamp based on camera's FPS.
            fly["timestamp"] = np.round(
                np.arange(fly.shape[0]) / FPS, decimal_precision)

            count += 1

    flies = pd.concat(flies_dict.values(), ignore_index=True)

    return flies
