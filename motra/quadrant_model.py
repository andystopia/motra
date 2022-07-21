import motra_model
from dataclasses import dataclass


@dataclass(frozen=True)
class QuadrantModel:
    top_left: motra_model.MotraModel
    top_right: motra_model.MotraModel
    bottom_right: motra_model.MotraModel
    bottom_left: motra_model.MotraModel
