
# from ....src.consts import MAP_RANGE, GRID_RANGE #TODO how to import here

##############################TODO remove##############################################
# Map Range

MAP_RANGE = {
    "LAT": {
        "MIN": 22,  # degrees N
        "MAX": 53,  # degrees N
    },
    "LON": {
        "MIN": -131,  # degrees E
        "MAX": -66,  # degrees E
    },
    "ALT": {
        "MIN": 0,  # ft
        "MAX": 45_000,  # ft
    },
}

# Grid Range

GRID_RANGE = {
    "LAT": 1500, # number of rows
    "LON": 2500, # number of cols
    "ALT": 91,   # number of z layers
}

############################################################################

import numpy as np
from matplotlib import pyplot as plt


# Chosen as shown for now, #TODO decide if this is any good
BACKGROUND_RISK = 0.0001 # smallest lookup turb risk = .1, smallest scale value for that is 1% so .001, then we said 10% of that


# Conversion constants for converting grid cells to KM
KM_PER_NM = 1.852
NM_PER_DEGLAT = 60 
NM_PER_DEGLON_EQUATOR = 60

# In the middle of our range to minimize distortional effect
NM_PER_DEGLON_US      = NM_PER_DEGLON_EQUATOR * np.cos(((MAP_RANGE["LAT"]["MAX"] + MAP_RANGE["LAT"]["MIN"]) / 2) * (np.pi / 180))

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

# These values are relative to the background risk
NEG_RISKS_RADIAL = [[0,   2,    5,    10,   15,   30,   100], 
             [0.89,0.91, 0.93, 0.95, 0.97, 0.99, 1.0]]

# These values are relative to the risk value of the pirep
# The data below is organized as [[dist from source], [risk]] and comes
# from the source above, with the bounding zones added in by hand
SEV_RISKS_RADIAL = np.array([[0, 30, 35, 45, 50, 60, 95, 130], 
                      [1.0, 0.75, 0.35, 0.15, 0.075, 0.035, 0.015, 0.0]])
MOD_RISKS_RADIAL = np.array([[0,  20, 30, 35,  50,  100, 120],
                      [0.75, 0.35, 0.15, 0.075, 0.035, 0.015, 0.0]])
LGT_RISKS_RADIAL = np.array([[0, 10, 15, 25, 40, 95, 110],
                      [0.75, 0.35, 0.15, 0.075, 0.035, 0.015, 0.0]])

# Generates an empty grid representing an area of effect that we should use to
# spread a pirep in the horzintal direction
def create_empty_grid(bounding):
    return np.full(((2 * int(np.ceil((100 + bounding) * KM_TO_CELLS_VERTICAL))) | 1, 
                     (2 * int(np.ceil((100 + bounding) * KM_TO_CELLS_HORIZONTAL))) | 1), np.nan)
def calc_risk_from_dist(dist_km, risks):
    idx = np.abs(risks[0] - dist_km).argmin()
    if risks[0][idx] > dist_km:
        idx = idx - 1
    return risks[1][idx]

# Creates the radial component of the factors that we spread a priep's risk factor by
# will be applied to a pirerp to approximate the area of effect of an instance of
# reported turbulence 
def create_radial_grid(severity, risks):
    grid = create_empty_grid(BOUNDING_ZONES[severity])
    height, width = grid.shape
    center_y = height // 2 + 1
    center_x = width  // 2 + 1
    print(severity, grid.shape)
    for y in range(0, height):
        for x in range(0, width):
            try:
                x_km = (x - center_x) / KM_TO_CELLS_HORIZONTAL 
                y_km = (y - center_y) / KM_TO_CELLS_VERTICAL
                
                
                dist = np.sqrt(x_km**2 + y_km**2)
                grid[x][y] = calc_risk_from_dist(dist, risks)
                
                # Ensures the actual location of the source risk stay the same
                # or is set to 0 for NEG event
                if ((x, y) == (center_x, center_y)):
                    if severity == 'NEG':
                        grid[x][y] = 0
                    else:
                        grid[x][y] = 1
            except:
                print(x, y)
    return grid

fig, axes = plt.subplots(1, 4, figsize=(12, 4))

# List of grids and titles
grids = [create_radial_grid('NEG', NEG_RISKS_RADIAL), create_radial_grid("LGT", LGT_RISKS_RADIAL), create_radial_grid('MOD', MOD_RISKS_RADIAL), create_radial_grid('SEV', SEV_RISKS_RADIAL)]
titles = ["NEG", "LGT", "MOD", "SEV"]

# Loop through subplots and plot each grid
for ax, grid, title in zip(axes, grids, titles):
    ax.imshow(grid, cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")  # Remove axes for a clean look

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("three_grids_side_by_side.png", dpi=300, bbox_inches="tight")

# Show the figure
plt.show()