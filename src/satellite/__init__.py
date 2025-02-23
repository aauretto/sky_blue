import numpy as np
import datetime as dt
from goes2go import GOES
from xarray import Dataset, DataArray
import numpy.typing as npt


def fetch(timestamp: dt.datetime, satellite: GOES) -> Dataset:
    return satellite.nearesttime(timestamp, download=False)


def fetch_range(start: dt.datetime, end: dt.datetime, satellite: GOES) -> Dataset:
    return satellite.timerange(start, end, return_as="xarray")


def fetch_bands(data: Dataset, bands: list[int]) -> DataArray:
    return (
        data[[f"CMI_C{band:02d}" for band in bands]]
        .to_dataarray(dim="band")
        .transpose("t", "y", "x", "band")
    )


# Code pulled from the NOAA Documentation for the AWS Bucket
def calculate_coordinates(data: Dataset) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    # Read in GOES ABI fixed grid projection variables and constants
    x_coordinate_1d = data.variables["x"][:]  # E/W scanning angle in radians
    y_coordinate_1d = data.variables["y"][:]  # N/S elevation angle in radians
    projection_info = data.variables["goes_imager_projection"]
    lon_origin = projection_info.attrs["longitude_of_projection_origin"]
    H = (
        projection_info.attrs["perspective_point_height"]
        + projection_info.attrs["semi_major_axis"]
    )
    r_eq = projection_info.attrs["semi_major_axis"]
    r_pol = projection_info.attrs["semi_minor_axis"]

    # Create 2D coordinate matrices from 1D coordinate vectors
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)

    # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
    np.seterr(all="ignore")

    # Equations to calculate latitude and longitude
    lambda_0 = (lon_origin * np.pi) / 180.0
    a_var = np.power(np.sin(x_coordinate_2d), 2.0) + (
        np.power(np.cos(x_coordinate_2d), 2.0)
        * (
            np.power(np.cos(y_coordinate_2d), 2.0)
            + (
                ((r_eq * r_eq) / (r_pol * r_pol))
                * np.power(np.sin(y_coordinate_2d), 2.0)
            )
        )
    )
    b_var = -2.0 * H * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    c_var = (H**2.0) - (r_eq**2.0)
    r_s = (-1.0 * b_var - np.sqrt((b_var**2) - (4.0 * a_var * c_var))) / (2.0 * a_var)
    s_x = r_s * np.cos(x_coordinate_2d) * np.cos(y_coordinate_2d)
    s_y = -r_s * np.sin(x_coordinate_2d)
    s_z = r_s * np.cos(x_coordinate_2d) * np.sin(y_coordinate_2d)

    abi_lat = (180.0 / np.pi) * (
        np.arctan(
            ((r_eq * r_eq) / (r_pol * r_pol))
            * (s_z / np.sqrt(((H - s_x) * (H - s_x)) + (s_y * s_y)))
        )
    )
    abi_lon = (lambda_0 - np.arctan(s_y / (H - s_x))) * (180.0 / np.pi)

    return abi_lat, abi_lon


def project(
    lat: npt.ArrayLike | npt.DTypeLike,
    lon: npt.ArrayLike | npt.DTypeLike,
    temps: npt.ArrayLike | npt.DTypeLike,
) -> npt.ArrayLike | npt.DTypeLike:
    from consts import MAP_RANGE, GRID_RANGE
    from utils.convert import convert_coord as convert
    print("Starting Projection")
    # Create coordinate mask
    lat_mask = (lat >= MAP_RANGE["LAT"]["MIN"]) & (lat <= MAP_RANGE["LAT"]["MAX"])
    lon_mask = (lon >= MAP_RANGE["LON"]["MIN"]) & (lon <= MAP_RANGE["LON"]["MAX"])
    mask = lat_mask & lon_mask

    # Map to grid coordinates
    rows = convert(lat[mask], "LAT")
    cols = convert(lon[mask], "LON")
    temps = temps[:, mask]

    # Shape data into grid
    data = np.zeros(
        (temps.shape[0], GRID_RANGE["LAT"], GRID_RANGE["LON"], temps.shape[-1])
    )
    data[:, rows, cols, :] = temps

    print("Returning Projection")
    return data

#TODO verify that the output is still correct now that we have mutli bands and multi files
def smooth(data: npt.ArrayLike | npt.DTypeLike) -> npt.ArrayLike | npt.DTypeLike:
    from scipy.ndimage import distance_transform_edt
    num_files = data.shape[0]

    for i in range(num_files):
        empty_mask = data[i] <= 0
        indices = distance_transform_edt(
            empty_mask, return_distances=False, return_indices=True
        )
        data[i][empty_mask] = data[i][tuple(indices[:, empty_mask])]

    return data


def union_sat_data(
    west: npt.ArrayLike | npt.DTypeLike, east: npt.ArrayLike | npt.DTypeLike
) -> npt.ArrayLike | npt.DTypeLike:
    pass
