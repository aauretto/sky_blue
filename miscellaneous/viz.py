from goes2go import GOES
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from goes2go.tools import abi_crs
import xarray as xr
from cmipc_data_retrieval import *
from netCDF4 import Dataset
import numpy as np
import pyproj
from noaa_calculate_degrees_func import calculate_degrees



projection = ccrs.AlbersEqualArea(
    central_longitude=-96,
    central_latitude=37.5,
    standard_parallels=(29.5, 45.5)
)

fig = plt.figure(figsize=(15, 12))
ax_east = fig.add_subplot(1, 2, 1, projection=projection)
ax_east_2 = fig.add_subplot(1, 2, 2, projection=projection)


data = retrieve_s3_data(get_range_CMIPC_data_fileKeys(2024, 311, 23, 59, 14)[0])
print("**************************************************************************************************")
print(type(data))
lat,lon = calculate_degrees(data)
print(f"{type(lat)=}")
dataset = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
print(type(dataset))
print("**************************************************************************************************")

crs, x, y = abi_crs(dataset, reference_variable="CMI")
print(f"{len(x) = }")
# print(crs)
# print(dir(crs))
# print(type(crs))
ax_east.coastlines(resolution="50m")

print(type(x))

latData = np.ma.getdata(lat)
latMask = np.ma.getmaskarray(lat)
latData[latMask] = 100

lonData = np.ma.getdata(lon)
lonMask = np.ma.getmaskarray(lon)
lonData[lonMask] = 100

ax_east.pcolormesh(x, y, data["CMI"], transform=crs)

ax_east_2.coastlines(resolution="50m")
# print(len(lat))
# print("Dir of lat is ", dir(lat))
# print(type(lat))
ax_east_2.pcolormesh(lonData, latData, data["CMI"])




print(f"{lon.shape = }")
print(lon)


plt.show()

# nc_dataset = Dataset('in_memory.nc', mode='r', memory=data_io.read())
# noaa_orig = Dataset("./goes16_abi_conus_lat_lon/goes16_abi_conus_lat_lon.nc", 'r')
# noaa_interpolated = Dataset("./goes16_abi_conus_interpolated_lat_lon/goes16_abi_conus_interpolated_lat_lon.nc", 'r')
# noaa_orig = noaa_orig.variables["latitude"][:]
# noaa_interpolated = noaa_interpolated.variables["latitude"][:]
# print(noaa_orig[np.ma.getmaskarray(noaa_orig)])
# print(noaa_interpolated[np.ma.getmaskarray(noaa_orig)])


# fig = plt.figure(figsize=(15, 12))
# ax_east = fig.add_subplot(1, 2, 2, projection=projection)

# for (ax, sat) in zip([ax_east], [sat_east]):
#     data = sat.nearesttime("2024-11-06 23:59:00")
#     crs, x, y = abi_crs(data, reference_variable="CMI_C13")

#     ax.coastlines(resolution="50m")
#     ax.pcolormesh(x, y, data.rgb.WaterVapor(), transform=crs)

# plt.show()