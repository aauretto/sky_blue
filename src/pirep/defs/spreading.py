"""
File: spreading.py

Description: A file containing utilities to spread PIREPs into risk areas that
             represent turbulence occurances.
             
Usage: 
    Call concatenate_all_pireps with a list of pireps report objects to generate
    a map of CONUS with turbulence risk values representing the risk of encountering
    turbulence determined from the provided PIREPs.
"""

import numpy as np
import numpy.typing as npt
from scipy import constants as u
from pirep.defs.aircraft import Aircraft
from pirep import PirepGrid

from consts import MAP_RANGE, GRID_RANGE

import pirep.defs.merge as merge
from Logger import LOGGER

from pirep.defs.generate_spreading_arrays import create_radial_grid

# Disks that represent risk of turbulence relative to the point the pirep was reported at.
RADIAL_KERNELS = {
    "NEG": create_radial_grid("NEG"),
    "LGT": create_radial_grid("LGT"),
    "MOD": create_radial_grid("MOD"),
    "SEV": create_radial_grid("SEV"),
}

# Arrays that indicate risk of turbulence at a given altitude level as a 
# factor of the risk value for that turbulence event.
# The first array represents distance from center altitude level in km and the
# second represents the factor mentioned. 
ALT_RISKS = {
    "NEG": np.array(
        [
            [-4, -3.3, -2.5, -2, -1.5, 0, 1, 1.5, 2.25, 3.5, 6, np.inf],
            [
                np.nan,
                0.97,
                0.95,
                0.93,
                0.91,
                0.89,
                0.91,
                0.93,
                0.95,
                0.97,
                0.99,
                np.nan,
            ],
        ]
    ),
    "LGT": np.array(
        [
            [-4, -3, -1.6, 0.6, 2, 3.6, 6, np.inf],
            [np.nan, 0.075, 0.15, 0.75, 0.15, 0.075, 0.035, np.nan],
        ]
    ),
    "MOD": np.array(
        [
            [-4, -3, 1.5, 2.75, 4, 6, np.inf],
            [np.nan, 0.15, 0.35, 0.15, 0.075, 0.035, np.nan],
        ]
    ),
    "SEV": np.array(
        [
            [-4, -3.5, -1, -0.5, 1.25, 2.5, 3.5, 4.5, 6, np.inf],
            [np.nan, 0.35, 0.75, 1.0, 0.75, 0.35, 0.15, 0.075, 0.035, np.nan],
        ]
    ),
}

# Conversion from aircraft size + intensity to a unified intensity value
PIREP_INT_CRAFT_TO_SPREAD_INT = {
    Aircraft.LGT: {"NEG": "NEG", "LGT": "LGT", "MOD": "MOD", "SEV": "MOD"},
    Aircraft.MED: {"NEG": "LGT", "LGT": "MOD", "MOD": "MOD", "SEV": "SEV"},
    Aircraft.HVY: {"NEG": "MOD", "LGT": "SEV", "MOD": "SEV", "SEV": "SEV"},
}

def spread_pirep(
    grid            : npt.NDArray, 
    aircraft        : Aircraft,
    intensity       : str, 
    BACKGROUND_RISK : float):
    """
    Wrapper so we can call spread_pirep instead of both vert then horiz.

    Parameters
    ----------
    grid: npt.NDArray
        3D Array representing a single turbulence event
    aircraft: aircraft
        Aircraft size provided by Pirep.defs.aircraft
    intensity: str
        Intensity of the reported event
    BACKGROUND_RISK: float
        Base risk of turbulence 

    Returns
    -------
    None

    Notes 
    -----
    Modifies grid in place
    """
    intensity = PIREP_INT_CRAFT_TO_SPREAD_INT[aircraft][intensity]
    vertical_spread(grid, intensity, BACKGROUND_RISK)
    radial_spread(grid, intensity, BACKGROUND_RISK)


# Expects a grid of shape (GRID_RANGE["LAT"], GRID_RANGE["LON"], GRID_RANGE["ALT"])
# where every cell is np.nan except a single vertical column
def vertical_spread(
    grid            : npt.NDArray,
    intensity       : str,
    BACKGROUND_RISK : float):
    """
    Spreads a single turbulence event in the vertical direction

    Parameters
    ----------
    grid: npt.NDArray
        3D Array representing a single turbulence event
    intensity: str
        Intensity of the reported event
    BACKGROUND_RISK: float
        Base risk of turbulence 

    Returns
    -------
    None

    Notes 
    -----
    Modifies grid in place
    """
    # Get indicies where grid is not NaN
    vals = np.argwhere(~np.isnan(grid))

    # Need to make sure everything is in order to get min and max

    vals = vals[vals[:, 2].argsort()]
    lat, lon = vals[0][:2]
    alt_min_idx = vals[0][-1]
    alt_max_idx = vals[-1][-1]

    base_risk = grid[lat][lon][alt_min_idx]
    if intensity == "NEG":
        base_risk = BACKGROUND_RISK

    for i in range(len(MAP_RANGE["ALT"]["RANGE"])):
        if i < alt_min_idx:  # Spread down
            # i => (ft -> km) => Risk (from table, snap to ceil)
            dist_km = (
                MAP_RANGE["ALT"]["RANGE"][i] - MAP_RANGE["ALT"]["RANGE"][alt_min_idx]
            ) * (u.foot / u.kilo)

        elif i > alt_max_idx:  # Spread up
            dist_km = (
                MAP_RANGE["ALT"]["RANGE"][i] - MAP_RANGE["ALT"]["RANGE"][alt_max_idx]
            ) * (u.foot / u.kilo)
        else:  # In alt band contained in pirep, dont modify turb risk
            continue
        idx = np.abs(ALT_RISKS[intensity][0] - dist_km).argmin()
        if ALT_RISKS[intensity][0][idx] < dist_km:
            idx = idx + 1

        grid[lat][lon][i] = ALT_RISKS[intensity][1][idx] * base_risk


def radial_spread(
    grid            : npt.NDArray, 
    intensity       : str, 
    BACKGROUND_RISK : float):
    """
    Spreads a single turbulence event in the horizontal direction

    Parameters
    ----------
    grid: npt.NDArray
        3D Array representing a single turbulence event
    intensity: str
        Intensity of the reported event
    BACKGROUND_RISK: float
        Base risk of turbulence 

    Returns
    -------
    None

    Notes 
    -----
    Modifies grid in place
    """

    kernel = RADIAL_KERNELS[intensity]
    vals = np.argwhere(~np.isnan(grid))
    # All vals are in a vertical column so pos of pirep in lat,lon is same across all vals
    lat, lon, _ = vals[0]

    # max sizes of axes for grid we are modifying and kernel we are applying to it
    g_lat_shp, g_lon_shp, g_alt_shp = grid.shape
    k_lat_shp, k_lon_shp = kernel.shape

    k_lat_center, k_lon_center = k_lat_shp // 2, k_lon_shp // 2

    # Slicing bounds on the grid
    g_lat_min = max(0, lat - k_lat_shp // 2)
    g_lat_max = min(g_lat_shp, lat + k_lat_shp // 2 + 1)
    g_lon_min = max(0, lon - k_lon_shp // 2)
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
        if intensity == "NEG":
            risk = BACKGROUND_RISK
        grid[g_lat_min:g_lat_max, g_lon_min:g_lon_max, alt] = (
            risk * kernel[k_lat_min:k_lat_max, k_lon_min:k_lon_max]
        )


# Grid should be a slice of the larger grid equal to kernel size
def add_pirep(
    grid            : npt.NDArray, 
    prData          : PirepGrid, 
    aircraft        : Aircraft, 
    intensity       : str, 
    BACKGROUND_RISK : float):
    """
    Adds a pirep represented by prData to grid

    Parameters
    ----------
    grid: npt.NDArray
        3D Array representing turbulence events in CONUS
    prData: PirepGrid
        PirepGrid struct representing a single PIREP
    aircraft: Aircraft
        Size class of the aircraft that reported the given PIREP
    intensity: str
        Intensity of the reported event
    BACKGROUND_RISK: float
        Base risk of turbulence 

    Returns
    -------
    None

    Notes 
    -----
    Modifies grid in place
    """

    # Localize data from prData
    lat      = prData.lat_idx
    lon      = prData.lon_idx
    alt_min  = prData.alt_min_idx
    alt_max  = prData.alt_max_idx
    turb_idx = prData.turbulence_idx

    # Compute kernel, put it on orig grid using vector crap
    kernel = RADIAL_KERNELS[PIREP_INT_CRAFT_TO_SPREAD_INT[aircraft][intensity]]

    # max sizes of axes for grid we are modifying and kernel we are applying to it
    g_lat_shp, g_lon_shp, g_alt_shp = grid.shape
    k_lat_shp, k_lon_shp = kernel.shape

    # Subgrid = area we project pirep into
    subGrid = np.full(shape=(k_lat_shp, k_lon_shp, GRID_RANGE["ALT"]), fill_value = np.nan)

    # Put turb index in middle of kernel we want to spread
    subGrid[subGrid.shape[0] // 2, subGrid.shape[0] // 2, alt_min : alt_max] = turb_idx

    spread_pirep(subGrid, aircraft, intensity, BACKGROUND_RISK)

    k_lat_center, k_lon_center = k_lat_shp // 2, k_lon_shp // 2

    # Slicing bounds on the grid
    g_lat_min = max(0, lat - k_lat_shp // 2)
    g_lat_max = min(g_lat_shp, lat + k_lat_shp // 2 + 1)
    g_lon_min = max(0, lon - k_lon_shp // 2)
    g_lon_max = min(g_lon_shp, lon + k_lon_shp // 2 + 1)

    # Slicing bounds for kernel
    k_lat_min = k_lat_center - (lat - g_lat_min)
    k_lat_max = k_lat_center + (g_lat_max - lat)
    k_lon_min = k_lon_center - (lon - g_lon_min)
    k_lon_max = k_lon_center + (g_lon_max - lon)

    target_area = grid[g_lat_min : g_lat_max, g_lon_min : g_lon_max , :]
    kernel_area = subGrid[k_lat_min : k_lat_max, k_lon_min : k_lon_max , :]
    grid[g_lat_min : g_lat_max, g_lon_min : g_lon_max , :] = np.fmax(target_area, kernel_area)

# Function that takes reports and spreads all PIREPS and smooshes everything together iteratively
def concatenate_all_pireps(reports: list[dict], background_risk: float):
    """
    Generates a grid representing a map of CONUS denoting risk of turbulence
    in a given area.

    Parameters
    ----------
    reports: list[dict]
        List of PIREPs reported over CONUS where turbulence was reported.
    background_risk: float
        The base risk of encountering turbulence

    Returns
    -------
    finalGrid: npt.NDarray
        A grid that represents turbulence risk over the entirity of CONUS
    """
    # Make final grid to spread all events onto
    finalGrid = np.full(
        (GRID_RANGE["LAT"], GRID_RANGE["LON"], GRID_RANGE["ALT"]), np.nan,
        dtype=np.float32
    )

    # Add all pireps to the grid one by one
    import pirep as pr
    for report in reports:
        prGridData, aircraft, intensity = pr.compute_grid(report)
        # Add targeted pirep to grid
        try:
            add_pirep(finalGrid, prGridData, aircraft, intensity, background_risk)
        except Exception:
            LOGGER.error(f"Failed to add pirep {report}\n", exc_info=True)

    # Fill in bg risk everywhere we didnt previously fill in
    mask = np.isnan(finalGrid) | (finalGrid == -np.inf)
    finalGrid[mask] = np.random.uniform(
        1e-5, 1e-7, size=mask.sum()
    )  # TODO document magic numbers

    return finalGrid
