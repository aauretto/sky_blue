"""
File: generate_spreading_arrays.py

Description: A file containing utilities to create spreading factors for PIREPs.
             
Usage: 
    Call create_radial_grid with some turbulence intensity value 
    ("NEG", "LGT", "MOD", "SEV") to obtain a 2d array that contains 
    weights for spreading a PIREP of that intensity.
"""

import numpy as np
from consts import GRID_RANGE, MAP_RANGE
import numpy.typing as npt

### Conversion constants for converting grid cells to KM and back ###

# Standard conversion factors for associated units
KM_PER_NM             = 1.852
NM_PER_DEGLAT         = 60 
NM_PER_DEGLON_EQUATOR = 60

# Approximate distance in NM per Deg of Longitude in the middle of CONUS
NM_PER_DEGLON_US = NM_PER_DEGLON_EQUATOR * np.cos(((MAP_RANGE["LAT"]["MAX"] + MAP_RANGE["LAT"]["MIN"]) / 2) * (np.pi / 180))

# Dimensions of the map we project onto in KM
HEIGHT_US_KM = KM_PER_NM * NM_PER_DEGLAT * (MAP_RANGE["LAT"]["MAX"] - MAP_RANGE["LAT"]["MIN"]) 
LENGTH_US_KM = KM_PER_NM * NM_PER_DEGLON_US * (MAP_RANGE["LON"]["MAX"] - MAP_RANGE["LON"]["MIN"])

# Conversion Factors 
KM_TO_CELLS_VERTICAL = GRID_RANGE["LAT"] / HEIGHT_US_KM 
KM_TO_CELLS_HORIZONTAL = GRID_RANGE["LON"] / LENGTH_US_KM

### Radial Computations ###
# The below dictionary described risk as a function of distance from the center
# of a turbulence event. 
#
#  Numbers used were chosen by hand and based on this source:
# https://journals.ametsoc.org/view/journals/bams/aop/BAMS-D-23-0142.1/BAMS-D-23-0142.1.pdf
#
# The paper uses a 10dBZ boundary so we implemented these bounding boxes
# which use the risk classes on page 14
BOUNDING_ZONES = {
    "NEG" : 0,
    "LGT" : 10,
    "MOD" : 20,
    "SEV" : 30
}

# The data below is organized as [[dist from source], [risk]]
RADIAL_RISKS = {
# These values are relative to the background risk
"NEG" : [[0,   2,    5,    10,   15,   30,   100], 
             [0.89,0.91, 0.93, 0.95, 0.97, 0.99, np.nan]],

# These values are relative to the risk value of the pirep
"LGT" : np.array([[0, 10, 15, 25, 40, 95, 110],
                  [0.75, 0.35, 0.15, 0.075, 0.035, 0.015, np.nan]]),
"MOD" : np.array([[0,  20, 30, 35,  50,  100, 120],
                  [0.75, 0.35, 0.15, 0.075, 0.035, 0.015, np.nan]]),
"SEV" : np.array([[0, 30, 35, 45, 50, 60, 95, 130], 
                  [1.0, 0.75, 0.35, 0.15, 0.075, 0.035, 0.015, np.nan]])
}


# Generates an empty grid representing an area of effect that we should use to
# spread a pirep in the horzintal direction
def create_empty_grid(bounding: int):
    """
    Creates empty PIREP spreading grid to be populated with risk values. 

    Parameters
    ----------
    bounding: int
        Extra padding to add to the grid we create.
    Returns
    -------
    : npt.NDarray
        A grid to be filled in with turbulence risk values. Dtype is float32
        and all values are initialized to np.nan 
    """
    # Create and return grid of specified size. Bitwise or '|' used to ensure
    # grid is odd and has a center cell
    return np.full(((2 * int(np.ceil((100 + bounding) * KM_TO_CELLS_VERTICAL))) | 1,  
                     (2 * int(np.ceil((100 + bounding) * KM_TO_CELLS_HORIZONTAL))) | 1),
                     np.nan, dtype=np.float32)

def calc_risk_from_dist(distKm : float, risks : npt.NDArray):
    """
    Determines the risk value distKm kilometers from the center of a turbulence
    event

    Parameters
    ----------
    distKm: float
        A distance in km from the center of a turbulence event
    risks: npt.NDarray
        A 2D array mapping intensities to risk values. Must be an entry in RADIAL_RISKS
        defined in this file.
    Returns
    -------
    : float
        Risk value at the specified distance from a turbulence event.       
    """
    idx = np.abs(risks[0] - distKm).argmin()
    if risks[0][idx] > distKm:
        idx = idx - 1
    return risks[1][idx]

# Creates the radial component of the factors that we spread a priep's risk factor by
# will be applied to a pirerp to approximate the area of effect of an instance of
# reported turbulence 
def create_radial_grid(intensity : str):
    """
    Generates a grid that has the weights used to spread a turbulence event.

    Parameters
    ----------
    intensity : str
        The intensty of the desired spreading grid to create. Must be "NEG", 
        "LGT", "MOD", or "SEV"
    
    Returns
    -------
    grid: npt.NDArray
        The grid containing weights used to spread a PIREP
    """
    # Set up empty grid 
    risks = RADIAL_RISKS[intensity]
    grid = create_empty_grid(BOUNDING_ZONES[intensity])
    height, width = grid.shape
    center_y = height // 2 + 1
    center_x = width  // 2 + 1

    # Populate grid with weights
    for y in range(0, height):
        for x in range(0, width):
            try:
                x_km = (x - center_x) / KM_TO_CELLS_HORIZONTAL 
                y_km = (y - center_y) / KM_TO_CELLS_VERTICAL
                
                dist = np.sqrt(x_km**2 + y_km**2)
                grid[x][y] = calc_risk_from_dist(dist, risks)
                
                # Ensures the actual location of the source risk stay the same
                if ((x, y) == (center_x, center_y)):
                    if intensity == 'NEG':
                        grid[x][y] = 0
                    else:
                        grid[x][y] = 1
            except Exception as e:
                raise RuntimeError(f"Failure to create radial spreading grid at point {x, y}. Error was:\n{e}")
    return grid
