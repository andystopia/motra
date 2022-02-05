import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def parse(path: str) -> pd.DataFrame:

    excel_file = pd.ExcelFile(path)
    flies = pd.DataFrame()

    count = 1
    for sheet_name in excel_file.sheet_names:
        fly = pd.read_excel(path, sheet_name=sheet_name)
        fly["fly_id"] = count
        fly["timestamp"] = np.round(np.arange(fly.shape[0]) / 30, 4)
        fly.sort_values(by=["fly_id", "timestamp"], inplace=True)
        flies = pd.concat([flies, fly], ignore_index=True)
        count += 1

    return flies
