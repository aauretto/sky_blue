import numpy as np

grid = np.full((5, 7), np.nan)
grid[0, 6] = 1


kernel = np.full((3, 3), 10)
vals = np.argwhere(~np.isnan(grid))
# All vals are in a vertical column so pos of pirep in lat,lon is same across all vals
lat, lon = vals[0] #TODO uncomment, _ = vals[0]

# max sizes of axes for grid we are modifying and kernel we are applying to it
g_lat_shp, g_lon_shp = grid.shape #TODO, g_alt_shp = grid.shape
k_lat_shp, k_lon_shp = kernel.shape

k_lat_center, k_lon_center = k_lat_shp // 2, k_lon_shp // 2

print(f"{k_lat_center=}, {k_lon_center=}")

# Slicing bounds on the grid
g_lat_min = max(0        , lat - k_lat_shp // 2)
g_lat_max = min(g_lat_shp, lat + k_lat_shp // 2 + 1)
g_lon_min = max(0        , lon - k_lon_shp // 2)
g_lon_max = min(g_lon_shp, lon + k_lon_shp // 2 + 1)
print(f"{g_lat_min=}, {g_lat_max=}, {g_lon_min=}, {g_lon_max=}")


# Slicing bounds for kernel
k_lat_min = k_lat_center - (lat - g_lat_min)
k_lat_max = k_lat_center + (g_lat_max - lat)
k_lon_min = k_lon_center - (lon - g_lon_min)
k_lon_max = k_lon_center + (g_lon_max - lon)
print(f"kernel: \n"
      f"{k_lat_min=}\n"                     
      f"{k_lat_max=}\n"                     
      f"{k_lon_min=}\n"           
      f"{k_lon_max=}")


print(f"Grid Shape = {grid.shape}")
print(f"Kernel Shape = {kernel.shape}")

# apply radial spread to each altitude level with a risk value in it
for val in vals:
    # alt = val[2] # TODO UNCOMMENT
    risk = grid[*val]
    # grid[g_lat_min:g_lat_max, g_lon_min:g_lon_max, alt] = risk * kernel[k_lat_min:k_lat_max, k_lon_min:k_lon_max] TODO: UNCOMMENT
    print(f" Grid Slice: {grid[g_lat_min:g_lat_max, g_lon_min:g_lon_max].shape}")
    print(f" Kernel Slice: {kernel[k_lat_min:k_lat_max, k_lon_min:k_lon_max].shape}")
    grid[g_lat_min:g_lat_max, g_lon_min:g_lon_max] = risk * kernel[k_lat_min:k_lat_max, k_lon_min:k_lon_max]
    
