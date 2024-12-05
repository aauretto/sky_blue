from goes2go import GOES
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from goes2go.tools import abi_crs
import xarray as xr
from cmipc_data_retrieval import *
import numpy as np

def calculate_degrees(file_id):
    
    # Read in GOES ABI fixed grid projection variables and constants
    x_coordinate_1d = file_id.variables['x'][:]  # E/W scanning angle in radians
    y_coordinate_1d = file_id.variables['y'][:]  # N/S elevation angle in radians
    projection_info = file_id.variables['goes_imager_projection']
    lon_origin = projection_info.longitude_of_projection_origin
    H = projection_info.perspective_point_height+projection_info.semi_major_axis
    r_eq = projection_info.semi_major_axis
    r_pol = projection_info.semi_minor_axis
    
    # Create 2D coordinate matrices from 1D coordinate vectors
    x_coordinate_2d, y_coordinate_2d = np.meshgrid(x_coordinate_1d, y_coordinate_1d)
    
    # Equations to calculate latitude and longitude
    lambda_0 = (lon_origin*np.pi)/180.0  
    a_var = np.power(np.sin(x_coordinate_2d),2.0) + (np.power(np.cos(x_coordinate_2d),2.0)*(np.power(np.cos(y_coordinate_2d),2.0)+(((r_eq*r_eq)/(r_pol*r_pol))*np.power(np.sin(y_coordinate_2d),2.0))))
    b_var = -2.0*H*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    c_var = (H**2.0)-(r_eq**2.0)
    r_s = (-1.0*b_var - np.sqrt((b_var**2)-(4.0*a_var*c_var)))/(2.0*a_var)
    s_x = r_s*np.cos(x_coordinate_2d)*np.cos(y_coordinate_2d)
    s_y = - r_s*np.sin(x_coordinate_2d)
    s_z = r_s*np.cos(x_coordinate_2d)*np.sin(y_coordinate_2d)
    
    # Ignore numpy errors for sqrt of negative number; occurs for GOES-16 ABI CONUS sector data
    np.seterr(all='ignore')
    
    abi_lat = (180.0/np.pi)*(np.arctan(((r_eq*r_eq)/(r_pol*r_pol))*((s_z/np.sqrt(((H-s_x)*(H-s_x))+(s_y*s_y))))))
    abi_lon = (lambda_0 - np.arctan(s_y/(H-s_x)))*(180.0/np.pi)
    
    return abi_lat, abi_lon

projection = ccrs.AlbersEqualArea(
    central_longitude=-96,
    central_latitude=37.5,
    standard_parallels=(29.5, 45.5)
)

fig = plt.figure(figsize=(15, 12))
ax_east = fig.add_subplot(1, 2, 2, projection=projection)
ax_east_2 = fig.add_subplot(1, 2, 2, projection=projection)


data = retrieve_s3_data(get_range_CMIPC_data_fileKeys(2024, 311, 23, 59, 14)[0])
print("**************************************************************************************************")
print(type(data))
lat,lon = calculate_degrees(data)
dataset = xr.open_dataset(xr.backends.NetCDF4DataStore(data))
print(type(dataset))
print("**************************************************************************************************")

crs, x, y = abi_crs(dataset, reference_variable="CMI")
print(len(x))
print(crs)
print(dir(crs))
print(type(crs))
ax_east.coastlines(resolution="50m")
ax_east.pcolormesh(x, y, data["CMI"], transform=crs)

ax_east_2.coastlines(resolution="50m")
print(len(lat))
ax_east_2.pcolormesh(lat, lon, data["CMI"])
plt.show()

# fig = plt.figure(figsize=(15, 12))
# ax_east = fig.add_subplot(1, 2, 2, projection=projection)

# for (ax, sat) in zip([ax_east], [sat_east]):
#     data = sat.nearesttime("2024-11-06 23:59:00")
#     crs, x, y = abi_crs(data, reference_variable="CMI_C13")

#     ax.coastlines(resolution="50m")
#     ax.pcolormesh(x, y, data.rgb.WaterVapor(), transform=crs)

# plt.show()