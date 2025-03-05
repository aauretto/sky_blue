import numpy as np
from scipy import constants as u

from consts import MAP_RANGE, GRID_RANGE

import utils.merge as merge

# Data Source: https://journals.ametsoc.org/view/journals/bams/aop/BAMS-D-23-0142.1/BAMS-D-23-0142.1.pdf
### Altitudinal Computations ### Taken from the page 14 graphs of the source above
from pirep.defs.generate_spreading_arrays import create_radial_grid

# Disks that represent risk of turbulence relative to the point the pirep was reported at.
RADIAL_KERNELS = {
    'NEG' : create_radial_grid('NEG'),
    'LGT' : create_radial_grid('LGT'),
    'MOD' : create_radial_grid('MOD'),
    'SEV' : create_radial_grid("SEV")
}

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

# Wrapper so we can call spread_pirep instead of both vert then horiz
def spread_pirep(grid, intensity):
    vertical_spread(grid, intensity)
    radial_spread(grid, intensity)



# Expects a grid of shape (GRID_RANGE["LAT"], GRID_RANGE["LON"], GRID_RANGE["ALT"])
# where every cell is np.nan except a single vertical column
def vertical_spread(grid, intensity):
    # Get indicies where grid is not NaN
    vals = np.argwhere(~np.isnan(grid))

    # Need to make sure everything is in order to get min and max
    
    # TODO: Before we get here make sure we drop all pireps that arent there
    vals = vals[vals[:, 2].argsort()]
    lat, lon = vals[0][:2]
    alt_min_idx = vals[0][-1]
    alt_max_idx = vals[-1][-1]

    base_risk = grid[lat][lon][alt_min_idx]

    for i in range(len(MAP_RANGE["ALT"]["RANGE"])):
        if i < alt_min_idx:   # Spread down
            # i => (ft -> km) => Risk (from table, snap to ceil)
            dist_km = (MAP_RANGE["ALT"]["RANGE"][i] - MAP_RANGE["ALT"]["RANGE"][alt_min_idx]) * (u.foot / u.kilo)

        elif i > alt_max_idx: # Spread up
            dist_km = (MAP_RANGE["ALT"]["RANGE"][i] - MAP_RANGE["ALT"]["RANGE"][alt_max_idx]) * (u.foot / u.kilo)
        else:                 # In alt band contained in pirep, dont modify turb risk
            continue
        idx = np.abs(ALT_RISKS[intensity][0] - dist_km).argmin()
        if ALT_RISKS[intensity][0][idx] < dist_km:
            idx = idx + 1
        
        grid[lat][lon][i] = ALT_RISKS[intensity][1][idx] * base_risk



def radial_spread(grid, intensity):
    kernel = RADIAL_KERNELS[intensity]
    vals = np.argwhere(~np.isnan(grid))
    # All vals are in a vertical column so pos of pirep in lat,lon is same across all vals
    lat, lon, _ = vals[0]

    # max sizes of axes for grid we are modifying and kernel we are applying to it
    g_lat_shp, g_lon_shp, g_alt_shp = grid.shape
    k_lat_shp, k_lon_shp = kernel.shape

    k_lat_center, k_lon_center = k_lat_shp // 2, k_lon_shp // 2

    
    # Slicing bounds on the grid
    g_lat_min = max(0        , lat - k_lat_shp // 2)
    g_lat_max = min(g_lat_shp, lat + k_lat_shp // 2 + 1)
    g_lon_min = max(0        , lon - k_lon_shp // 2)
    g_lon_max = min(g_lon_shp, lon + k_lon_shp // 2 + 1)

    # Slicing bounds for kernel
    k_lat_min = k_lat_center - (lat - g_lat_min)
    k_lat_max = k_lat_center + (g_lat_max - lat)
    k_lon_min = k_lon_center - (lon - g_lon_min)
    k_lon_max = k_lon_center + (g_lon_max - lon)

    # apply radial spread to each altitude level with a risk value in it
    for val in vals:
        alt = val[2]
        risk = grid[*val]
        grid[g_lat_min:g_lat_max, g_lon_min:g_lon_max, alt] = risk * kernel[k_lat_min:k_lat_max, k_lon_min:k_lon_max]
        
# Function that takes reports and spreads all PIREPS and smooshes everything together iteratively
# TODO: Handle negative turbulence case
def concatenate_all_pireps(reports):
    
    # make final grid and temp grid
    finalGrid = np.full((GRID_RANGE["LAT"], GRID_RANGE["LON"], GRID_RANGE["ALT"]), np.nan)

    import pirep as pr
    for row in range (len(reports)):
        print(f"{type(reports.iloc[row])=}\n{reports.iloc[row]}", "\n============================================================")
        tmpGrid, aircraft, intensity = pr.compute_grid(reports.iloc[row])

        # spread pirep in temp grid
        vertical_spread(tmpGrid, intensity)
        radial_spread(tmpGrid, intensity)

        # merge temp grid with final grid
        finalGrid = merge.merge_max([finalGrid, tmpGrid])

    return finalGrid
    

    