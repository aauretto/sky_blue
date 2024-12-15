# 51N/-131E, 
# 51N/-66E, 
# 23N/-66E, 
# 23N/-131E

import numpy as np
from pyproj import CRS, Transformer
from cmipc_data_retrieval import *
from noaa_calculate_degrees_func import calculate_degrees

# Define the Geographic CRS (WGS 84 - EPSG:4326)
geographic_crs = CRS.from_epsg(4326)

# Define the Albers Equal-Area Projection
# Example for Europe: central meridian at 10° longitude, and two standard parallels at 20°N and 50°N
albers_crs = CRS.from_proj4("+proj=aea +lat_1=23 +lat_2=51 +lon_0=-96 +datum=WGS84")

# Create a transformer
transformer = Transformer.from_crs(geographic_crs, albers_crs, always_xy=True)
lat_max =  51  # deg N
lat_min =  23  # deg N
lon_max = -66  # deg E
lon_min = -131 # deg E 

# Define the bounds for the grid - Transform lat and lon bounds
x_min, y_min = transformer.transform(lon_min, lat_min)
x_max, y_max = transformer.transform(lon_max, lat_max)

sat_data = retrieve_s3_data(get_range_CMIPC_data_fileKeys(2024, 311, 23, 59, 14)[0])


data_temps = sat_data.variables["CMI"][:]
lat, lon = calculate_degrees(sat_data)


# Transform latitudes and longitudes into Albers Equal-Area projection
x, y = transformer.transform(lon, lat)


# Define grid parameters (grid spacing and bounding box)
grid_spacing = 2000  # Grid cell size (2 km per cell)

# Make the grid we will dump everything into:
# Create the grid (x and y grid points)
x_grid = np.arange(x_min, x_max, grid_spacing)
y_grid = np.arange(y_min, y_max, grid_spacing)


# Create an empty data array (grid) with the same shape as the grid
data_array = np.zeros((len(y_grid), len(x_grid)))

print(f"{data_array.shape=}")
print(f"{x.shape=}")
print(f"{y.shape=}")
print(f"{data_temps.shape=}")


print(f"Xmin = {x_min}", f"Smallest transformed X = {x.min()}")

# Map the intensities to the grid
diffs = np.zeros(x.shape[1])
for i in range(x.shape[0]):
    print(i, end = ",", flush=True)
    for j in range(x.shape[1]):
        if not x.mask[i][j] and not y.mask[i][j] and not data_temps.mask[i][j]:
            row_idx = np.argmin(np.abs(y_grid - y[i][j]))
            # row_idx_2 = int(np.round((y[i][j] - y_min)/2000))
            # col_idx_2 = int(np.round((x[i][j] - x_min)/2000))
            col_idx = np.argmin(np.abs(x_grid - x[i][j]))
            data_array[row_idx, col_idx] = data_temps[i][j]
            # try:
            #     data_array[row_idx_2, col_idx_2] = data_temps[i][j] 
            # except:
            #     pass

            # diffs[j] = abs(row_idx - row_idx_2)
    # print(max(diffs))

fig = plt.figure(figsize=(15, 12))
ax_east = fig.add_subplot(1, 2, 2)
ax_east.pcolormesh(data_array)
plt.show()
