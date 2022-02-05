import pandas as pd
import numpy as np
from .fps import fps


def parse(path: str) -> pd.DataFrame:

    flies_dict = pd.read_excel(path, sheet_name=None)
    decimal_precision = 4

    count = 1
    for fly in flies_dict.values():
        if not fly.empty:
            fly["fly_id"] = count
            fly["timestamp"] = np.round(
                np.arange(fly.shape[0]) / fps, decimal_precision)
            count += 1

    flies = pd.concat(flies_dict.values(), ignore_index=True)

    return flies
