import numpy as np
import numpy.typing as npt

from consts import MAP_RANGE, GRID_RANGE


def convert_coord(
    values: npt.ArrayLike | npt.DTypeLike, axis: str
) -> npt.ArrayLike | npt.DTypeLike:
    # Normalize the latitude values to the range [0, 1]
    normalized = (values - MAP_RANGE[axis]["MIN"]) / (
        MAP_RANGE[axis]["MAX"] - MAP_RANGE[axis]["MIN"]
    )

    # Scale to the range [0, NUM_ROWS - 1]
    row_values = (normalized) * (GRID_RANGE[axis] - 1)

    # Round and convert to integers for indexing
    return np.round(row_values).astype(int)
