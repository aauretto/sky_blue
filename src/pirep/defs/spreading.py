import numpy as np
from consts import MAP_RANGE
from scipy import constants as u

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


ALSO we want to define three regions in latitude that each have their own unit conversion from DEG => KM to account for wonky we live on a sphere stuff

"""

# Data Source: https://journals.ametsoc.org/view/journals/bams/aop/BAMS-D-23-0142.1/BAMS-D-23-0142.1.pdf

### Altitudinal Computations ### Taken from the page 14 graphs of the source above

ALT_RISKS = {

"NEG" : np.array([
    [-4, -3.3, -2.5, -2, -1.5, 0, 1, 1.5, 2.25, 3.5, 6, np.inf],
    [np.nan, 0.97, 0.95, 0.93, 0.91, 0.89, 0.91, 0.93, 0.95, 0.97, 0.99, np.nan],
]),
"LGT" : np.array([
    [-4, -3, -1.6, 0.6, 2, 3.6, 6, np.inf],
    [np.nan, 0.075, 0.15, 0.75, 0.15, 0.075, 0.035, np.nan],
]),
"MOD" : np.array([
    [-4, -3, 1.5, 2.75, 4, 6, np.inf],
    [np.nan, 0.15, 0.35, 0.15, 0.075, 0.035, np.nan],
]),
"SEV" : np.array([
    [-4, -3.5, -1, -0.5, 1.25, 2.5, 3.5, 4.5, 6, np.inf],
    [np.nan, 0.35, 0.75, 1.0, 0.75, 0.35, 0.15, 0.075, 0.035, np.nan],
])
}

# Expects a grid of shape (GRID_RANGE["LAT"], GRID_RANGE["LON"], GRID_RANGE["ALT"])
# where every cell is np.nan except a single vertical column
def vertical_spread(grid, intensity):
    # Get indicies where grid is not NaN
    vals = np.argwhere(~np.isnan(grid))

    # Need to make sure everything is in order to get min and max
    vals = vals[vals[:, 2].argsort()]
    lat, lon = vals[0][:2]
    alt_min_idx = vals[0][-1]
    alt_max_idx = vals[-1][-1]

    base_risk = grid[lat][lon][alt_min_idx]

    for i in range(len(MAP_RANGE["ALT"]["RANGE"])):
        print(i, alt_min_idx, alt_max_idx)
        if i < alt_min_idx:   # Spread down
            # i => (ft -> km) => Risk (from table, snap to ceil)
            dist_km = (MAP_RANGE["ALT"]["RANGE"][i] - MAP_RANGE["ALT"]["RANGE"][alt_min_idx]) * (u.foot / u.kilo)

        elif i > alt_max_idx: # Spread up
            dist_km = (MAP_RANGE["ALT"]["RANGE"][i] - MAP_RANGE["ALT"]["RANGE"][alt_max_idx]) * (u.foot / u.kilo)
        else:                 # In alt band contained in pirep, dont modify turb risk
            continue
        print(f"{dist_km=} ")
        idx = np.abs(ALT_RISKS[intensity][0] - dist_km).argmin()
        if ALT_RISKS[intensity][0][idx] < dist_km:
            idx = idx + 1
        
        grid[lat][lon][i] = ALT_RISKS[intensity][1][idx] * base_risk
    print(grid[lat][lon][:])

