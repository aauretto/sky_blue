import numpy as np
import numpy.typing as npt


# Used to convert any raw lat/lon/alt coordinates (PIREPS, satellite) to the
# defined grid of lat/lon/alt values (in consts)
def convert_coord(
    values: npt.ArrayLike | npt.DTypeLike, axis: str
) -> npt.ArrayLike | npt.DTypeLike:
    from consts import GRID_RANGE, MAP_RANGE

    # Normalize the latitude values to the range [0, 1]
    normalized = (values - MAP_RANGE[axis]["MIN"]) / (
        MAP_RANGE[axis]["MAX"] - MAP_RANGE[axis]["MIN"]
    )

    # Scale to the range [0, NUM_ROWS - 1]
    row_values = (normalized) * (GRID_RANGE[axis] - 1)

    # Round and convert to integers for indexing
    return np.round(row_values).astype(int)
