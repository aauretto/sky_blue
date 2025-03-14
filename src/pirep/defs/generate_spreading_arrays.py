import numpy as np

from consts import GRID_RANGE, MAP_RANGE

# Conversion constants for converting grid cells to KM
KM_PER_NM = 1.852
NM_PER_DEGLAT = 60 
NM_PER_DEGLON_EQUATOR = 60

# In the middle of our range to minimize distortional effect
NM_PER_DEGLON_US = NM_PER_DEGLON_EQUATOR * np.cos(((MAP_RANGE["LAT"]["MAX"] + MAP_RANGE["LAT"]["MIN"]) / 2) * (np.pi / 180))

# Dims of our map in KM
HEIGHT_US_KM = KM_PER_NM * NM_PER_DEGLAT * (MAP_RANGE["LAT"]["MAX"] - MAP_RANGE["LAT"]["MIN"]) 
LENGTH_US_KM = KM_PER_NM * NM_PER_DEGLON_US * (MAP_RANGE["LON"]["MAX"] - MAP_RANGE["LON"]["MIN"])

# Unit conversions
KM_TO_CELLS_VERTICAL = GRID_RANGE["LAT"] / HEIGHT_US_KM 
KM_TO_CELLS_HORIZONTAL = GRID_RANGE["LON"] / LENGTH_US_KM

### Radial Computations ###

# The from the source uses a 10dBZ boundary so we implemented these bounding boxes
# which use the risk class (see charts on page 14) above the one at 0km
BOUNDING_ZONES = {
    "NEG" : 0,
    "LGT" : 10,
    "MOD" : 20,
    "SEV" : 30
}

# The data below is organized as [[dist from source], [risk]] and comes
# from the source above, with the bounding zones added in by hand
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
def create_empty_grid(bounding):
    return np.full(((2 * int(np.ceil((100 + bounding) * KM_TO_CELLS_VERTICAL))) | 1,  # | == Make it odd operator
                     (2 * int(np.ceil((100 + bounding) * KM_TO_CELLS_HORIZONTAL))) | 1),
                     np.nan, dtype=np.float32)

def calc_risk_from_dist(dist_km, risks):
    idx = np.abs(risks[0] - dist_km).argmin()
    if risks[0][idx] > dist_km:
        idx = idx - 1
    return risks[1][idx]

# Creates the radial component of the factors that we spread a priep's risk factor by
# will be applied to a pirerp to approximate the area of effect of an instance of
# reported turbulence 
def create_radial_grid(intensity):
    risks = RADIAL_RISKS[intensity]
    grid = create_empty_grid(BOUNDING_ZONES[intensity])
    height, width = grid.shape
    center_y = height // 2 + 1
    center_x = width  // 2 + 1
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
