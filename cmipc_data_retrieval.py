import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Function get_CMIPC_data_fileKeys
# Inputs:
#    Year     : int 2017-2024
#    Day      : int 1 - 366
#    Hour     : int 0 - 23
#    Band Num : int 1 - 16
#
# Returns:
#    dataset obj containing relevant GOES-16 data 

#               YEAR DAY HOUR                 BAND
#                 V   V   V                    V
## ABI-L2-CMIPC/2023/001/01/OR_ABI-L2-CMIPC-M6C14_G16_s20230010101173_e20230010103546_c20230010104092.nc EXAMPLE KEY

BUCKET_NAME = 'noaa-goes16'
def get_CMIPC_data_fileKeys(year, day, hour, bandNum):
    # Set up boto3 client that will access the bucket w/o authentication/creds.
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    # ensure we are only using CMIPC data (Cloud and Moisture Imagery Product for CONUS)
    prefix = 'ABI-L2-CMIPC/'
    
    # convert to yyyy/ddd/hh file prefix used to look up file in bucket
    prefix += f'{year}/{day:03}/{hour:02}'

    # ask bucket for all CMIPC data taken in hour hour on day day in year year
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    
    if "Contents" in response:
        fileList = []
        for object in response['Contents']:
            fileKey = object['Key']
            if f"C{bandNum:02}" in fileKey:
                print(f'Found file with key: {fileKey}')  # Print the file key
                fileList.append(fileKey)
        return fileList
            
# fileKey
#    fileKey  : string -- key of a specific file in the bucket
#    outFname : string -- location to store file on device
def retrieve_s3_data(fileKey, outFname = "my_example"):
    # Read the NetCDF data directly into an xarray Dataset
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # Open the file and read the data
    s3.download_file(BUCKET_NAME, fileKey, Filename=outFname)
    ds = Dataset(outFname, 'r')

    return ds

# show_CMIPC_image
#   ds : a netCDF4 Dataset containing CMIPC data
#
# TODO : visualization people, this is all you
def show_CMPIC_image(ds):
    CMI = ds.variables['CMI'][:]
    lon_origin = ds.variables['geospatial_lat_lon_extent'].geospatial_westbound_longitude
    lon_extent = ds.variables['geospatial_lat_lon_extent'].geospatial_eastbound_longitude
    lat_origin = ds.variables['geospatial_lat_lon_extent'].geospatial_southbound_latitude
    lat_extent = ds.variables['geospatial_lat_lon_extent'].geospatial_northbound_latitude

    fig = plt.figure(figsize=(8,8),dpi=200)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_origin, lon_extent, lat_origin, lat_extent], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    im = plt.imshow(CMI, origin='upper', extent=(lon_origin, lon_extent, lat_origin, lat_extent), transform=ccrs.PlateCarree())

    plt.show()

# show_CMIPC_from
# Shows a CMPIPC data thing from the specified time and in the specified band
#
# Inputs:
#   year     : int 2017 - 2024
#   day      : int 1 - 366
#   hour     : int 0 - 23
#   bandNum  : int 1 - 16
#   idx      : int               Represents the idx-th scan of the hour
#   tmpFname : string            the name of a file that we are temporarily making in the local directory to access the data
#
# NOTE: tmpFname MUST REALLY be different than any current filename in your directory
def show_CMIPC_from(year, day, hour, bandNum, idx=0, tmpFname="my_example"):
    fkeys = get_CMIPC_data_fileKeys(year, day, hour, bandNum)

    ds = retrieve_s3_data(fkeys[idx], outFname = tmpFname)

    show_CMPIC_image(ds)
    
    # Remove temp file
    ds.close()
    if os.path.exists(f'./{tmpFname}'):
        os.remove(f'./{tmpFname}')
        print(f"File '{tmpFname}' deleted successfully.")
    else:
        print(f"File '{tmpFname}' not found.")


def main():
    show_CMIPC_from(2023, 1, 1, 14) # TODO : have fun with different dates here

if __name__ == "__main__":
    main()