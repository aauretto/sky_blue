import numpy as np
from cmipc_data_retrieval import *
from noaa_calculate_degrees_func import calculate_degrees

NUM_COLS = 2500 # Number of columns (longitudes)
NUM_ROWS = 1500 # Number of rows (latitudes)

# The latitude and longitude bounds of CONUS, with some leeway in all directions
LON_MIN = -131
LON_MAX = -66
LAT_MIN = 23
LAT_MAX = 51

# Function lat_to_row
# Inputs:
#   latitudes : array-like    
# 
# Returns:
#   the row coordinates that correspond to the given latitudes

def lat_to_row(latitudes):
    # Normalize the latitude values to the range [0, 1]
    normalized_lat = (latitudes - LAT_MIN) / (LAT_MAX - LAT_MIN)

    # Scale to the range [0, NUM_ROWS - 1]
    row_values = (normalized_lat) * (NUM_ROWS - 1)

    return np.round(row_values).astype(int)  # Round and convert to integers for indexing

# Function lon_to_col
# Inputs:
#   longitudes : array-like    
# 
# Returns:
#   the col coordinates that correspond to the given longitudes

def lon_to_col(longitudes):
    # Normalize the longitude values to the range [0, 1]
    normalized_lon = (longitudes - LON_MIN) / (LON_MAX - LON_MIN)

    # Scale to the range [0, NUM_COLS - 1]
    col_values = normalized_lon * (NUM_COLS - 1)

    return np.round(col_values).astype(int)  # Round and convert to integers for indexing

# Function sat_data_mapped
# Inputs:
#   fileKey  : GOES file key
#   year     : int 2017-2024
#   day      : int 1 - 366   
#   hour     : int 0 - 23   
#   minute   : int 1 - 59   
#   bandNum  : int 1 - 16
#   data_var : the variable name of the data to be mapped
# Returns:
#   The satellite data mapped to a numpy array based on corresponding latitude and longitude values
# Notes: either fileKey or year-day-hour-minute-band must be provided
#        occasionally two points will map to the same cell in the array. These are ignored 
def sat_data_mapped(fileKey="", year=0, day=0, hour=0, minute=0, band=0, data_var = "CMI"):
    if fileKey == "":
        sat_data = retrieve_s3_data(get_range_CMIPC_data_fileKeys(year, day, hour, minute, band)[0])
    else:
        sat_data = retrieve_s3_data(fileKey)
    
    data_temps = sat_data.variables[data_var][:]
    lat, lon = calculate_degrees(sat_data)

    valid_lats_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
    valid_lons_mask = (lon >= LON_MIN) & (lon <= LON_MAX)
    valid_mask = valid_lats_mask & valid_lons_mask

    rows = lat_to_row(lat[valid_mask])
    cols = lon_to_col(lon[valid_mask])
    data_temps = data_temps[valid_mask]
    
    # This method ignores duplicate mappings
    mapped_data = np.zeros((NUM_ROWS, NUM_COLS))
    mapped_data[rows, cols] = data_temps

    # This method averages over duplicate mappings
    # sum_data = np.zeros((NUM_ROWS, NUM_COLS))
    # count_data = np.zeros((NUM_ROWS, NUM_COLS), dtype=int)
    #     # Accumulate the sum of values and count the occurrences for each grid point
    # for r, c, temp in zip(rows, cols, data_temps):
    #     sum_data[r, c] += temp
    #     count_data[r, c] += 1

    #     # Calculate the average where the count is greater than 0
    #     # Prevent division by zero by checking the count
    # mapped_data = np.where(count_data > 0, sum_data / count_data, 0)
    
    #TODO smoothing
    # from scipy.ndimage import gaussian_filter
    # smoothed_data = gaussian_filter(mapped_data, sigma=1)
    # mapped_data = np.where(mapped_data == 0, smoothed_data, mapped_data)

    return mapped_data

if __name__ == '__main__':
    fileKey = get_range_CMIPC_data_fileKeys(2024, 311, 23, 59, 14)[0]
    data_mapped = sat_data_mapped(fileKey)

    fig = plt.figure(figsize=(15, 12))
    ax_east = fig.add_subplot(1, 1, 1)
    ax_east.pcolormesh(data_mapped)
    plt.show()
