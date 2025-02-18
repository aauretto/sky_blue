"""
How are we gonna spread?

1 fx for each severity {NONE, LIGHT, MOD, SEVERE} based on colorful graphs from paper that WxExt sent us (in dis)

Each function takes:
    (grid , aircraft type , severity) => implies compute_grid needs to return this tuple or aircraft/severity needs to come from elsewhere.

That funciton:
1) looks into the grid for the point that is nonzero, ie contains the risk val from the lookup table in
   the locaiton the pirep was reported.
2) Define some mathematical curve that depends on distance from the src point (from (1)) in x and z
3) apply that curve to each point on the grid


ALSO we want to define three regions in latitude that each have their own unit conversion from DEG => KM to account for wonky we live on a sphere bitch stuff

"""
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

KM_PER_NM = 1.852
NM_PER_DEGLAT = 60 
NM_PER_DEGLON_EQUATOR = 60

# In the middle of our range to minimize distortional effect
NM_PER_DEGLON_US      = NM_PER_DEGLON_EQUATOR * np.cos(((MAP_RANGE["LAT"]["MAX"] + MAP_RANGE["LAT"]["MIN"]) / 2) * (np.pi / 180))


HEIGHT_US_KM = KM_PER_NM * NM_PER_DEGLAT * (MAP_RANGE["LAT"]["MAX"] - MAP_RANGE["LAT"]["MIN"]) 
LENGTH_US_KM = KM_PER_NM * NM_PER_DEGLON_US * (MAP_RANGE["LON"]["MAX"] - MAP_RANGE["LON"]["MIN"])

KM_TO_CELLS_VERTICAL = GRID_RANGE["LAT"] / HEIGHT_US_KM 
KM_TO_CELLS_HORIZONTAL = GRID_RANGE["LON"] / LENGTH_US_KM


# LGT_STEPS = []

# def spread(grid, aircraft, severity):
#     pass

