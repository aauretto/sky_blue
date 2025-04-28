

import datetime as dt
import numpy as np
import numpy.typing as npt
from goes2go import GOES
from xarray import DataArray, Dataset


def generate_timestamps(
    start: dt.datetime = dt.datetime(2017, 3, 1, 0, 3, tzinfo=dt.UTC),
    end: dt.datetime = dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.UTC),
) -> list[dt.datetime]:
    """
    Generates a list of 5 minutes seperated datetimes starting on minute 3
    of each year 2018-2024 and 2017 without Jan and Feb

    Parameters
    ----------
    start: dt.datetime
        The beginning of the time range to fetch
    end: dt.datetime
        The end of the time range to fetch
    
    Returns
    -------
    a list of datetimes in the range
    """

    timestamps = [[] for _ in range(12)]
    current_time = start

    while current_time < end:
        timestamps[current_time.minute // 5].append(current_time)
        current_time = current_time + dt.timedelta(minutes=5)

    return timestamps

def fetch(timestamp: dt.datetime, satellite: GOES) -> Dataset:
    """
    Gets a single satellite image nearest to timestamp

    Parameters
    ----------
    timestamp: dt.datetime
        The timestamp for the time we want an image for.
    satellite: GOES
        GOES object representing the satellite we want to fetch from
    
    Returns
    -------
    :Dataset
        xarray dataset containing a satellite image.
    """
    return satellite.nearesttime(
        timestamp.replace(tzinfo=None),
        return_as="xarray",
        download=False,
        verbose=False
    )


def fetch_range(start: dt.datetime, end: dt.datetime, satellite: GOES) -> Dataset:
    """
    Fetches all images in range [start, end]

    Parameters
    ----------
    start: dt.datetime
        The starting timestamp for the time we want an image for.
    end: dt.datetime
        The ending timestamp for the time we want an image for.
    satellite: GOES
        GOES object representing the satellite we want to fetch from
    
    Returns
    -------
    :Dataset
        xarray dataset containing all satellite images in range.

    """
    return satellite.timerange(start, end, return_as="xarray", download=False)


def fetch_bands(data: Dataset, bands: list[int]) -> DataArray:
    """
    Extracts bands of interest from data retrieved from satellite.

    Parameters
    ----------
    data: Dataset
        xarray dataset containing a satellite image.
    bands: list[int]
        Bands in data we want to keep.
    
    Returns
    -------
    :DataArray
        The input data but only keeping bands listed in bands
    """
    return (
        data[[f"CMI_C{band:02d}" for band in bands]]
        .to_dataarray(dim="band")
        .transpose("t", "y", "x", "band")
    )


def calculate_coordinates(data: Dataset) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """
    Takes a xarray dataset of a satellite image and returns the lat-lon data
    for each datapoint in the array.

    Code pulled from the NOAA Documentation for the AWS Bucket

    data: Dataset
        The input xarray dataset
    
    Returns
    -------
    :tuple[npt.ArrayLike, npt.ArrayLike]
        (Lon coords, Lat coords)
    """
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
    """
    Projects satellite image onto our grid with equal spaced lat and lon on x and y axes

    Parameters
    ----------
    lat: npt.ArrayLike | npt.DTypeLike
        Latitude coordinates
    lat: npt.ArrayLike | npt.DTypeLike
        Longitude coordinates
    lat: npt.ArrayLike | npt.DTypeLike
        Temprature values
    
    Returns
    -------
    :npt.ArrayLike | npt.DTypeLike
        A grid representing CONUS with data projected into the proper location
    """
    from consts import GRID_RANGE, MAP_RANGE
    from utils.convert import convert_coord as convert

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
        (temps.shape[0], GRID_RANGE["LAT"], GRID_RANGE["LON"], temps.shape[-1]),
        dtype=np.float32,
    )
    data[:, rows, cols, :] = temps

    return data


# Currently this smooths a single band in a single file *(MAY NOT WORK ON EVERY BAND)
# Expects 1500 x 2500
# want (F, 1500, 2500, b)
#   F - num files
#   B - num Bands
# Seems like when we get multiple bands we are smoothing across multiple bands :(
def smooth(all_data: npt.ArrayLike | npt.DTypeLike) -> npt.ArrayLike | npt.DTypeLike:
    """
    Fills in the gaps for all bands in the data. Due to projection, some pixels may
    not have values and need to be interpolated.

    all_data: npt.ArrayLike | npt.DTypeLike

    Returns
    -------
    :npt.ArrayLike | npt.DTypeLike
        Interpolated data.
    """
    n_files = all_data.shape[0]
    n_bands = all_data.shape[3]

    for f in range(n_files):
        for b in range(n_bands):
            all_data[f, :, :, b] = smooth_single_band(all_data[f, :, :, b])
    return all_data


def smooth_single_band(
    data: npt.ArrayLike | npt.DTypeLike,
) -> npt.ArrayLike | npt.DTypeLike:
    """
    Fills in the gaps for a single band of data

    Parameters
    ----------
    data: npt.ArrayLike
        A single band of pirep data
    
    Returns
    -------
    npt.ArrayLike
        The single band of pirep data smoothed
    """
    from scipy.ndimage import distance_transform_edt

    ## runs after projection, gets points that are in between buckets from
    #  projection AND things that are not inside frame of GOES E
    empty_mask = data <= 0  # If band data is lt 0 then it may not work
    indices = distance_transform_edt(
        empty_mask, return_distances=False, return_indices=True
    )
    data[empty_mask] = data[tuple(indices[:, empty_mask])]
    return data

