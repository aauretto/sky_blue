import os
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import io
from datetime import datetime, timezone, timedelta

# CONSTS
BUCKET_NAME = 'noaa-goes16'
MINUTE_INTERVAL = 5
SCAN_START_INDICATOR = "_s"
SCAN_START_INDICATOR_LENGTH = 2
START_MINUTE_OFFSET = 9
MINUTE_FIELD_WIDTH = 2

# Function get_range_CMIPC_data_fileKeys
# Inputs:
#   year    : int 2017-2024
#   day     : int 1 - 366   
#   hour    : int 0 - 23   
#   minute  : int 1 - 59   
#   bandNum : int 1 - 16   
# Returns:
#   All the scans in the interval of the given time +/- MINUTE_INTERVAL
# Notes: Will get keys within MINUTE_INTERVAL minutes from specified timestamp, rolling over or under as required

def get_range_CMIPC_data_fileKeys(year, day, hour, minute, bandNum):
    dt = datetime(year, month = 1, day = 1, tzinfo=timezone.utc) + timedelta(days=day-1, hours=hour, minutes=minute)
    
    # Over/underflow for minutes -- need to grab files from next/prev day, month, or year
    if minute < MINUTE_INTERVAL or minute + MINUTE_INTERVAL > 59:
        delta = timedelta(minutes= MINUTE_INTERVAL)
        lower_time = dt - delta
        lower_files = get_CMIPC_data_fileKeys(lower_time.year, lower_time.timetuple().tm_yday, lower_time.hour, bandNum, lower_time.minute, 59)

        upper_time = dt + delta
                                                                # We want to not include scans from the upper_time.minute-th minute hence the - 1
        upper_files = get_CMIPC_data_fileKeys(upper_time.year, upper_time.timetuple().tm_yday, upper_time.hour, bandNum, 0, upper_time.minute - 1)
        
        return lower_files + upper_files

    # No over/underflow
    return get_CMIPC_data_fileKeys(year, day, hour, bandNum, minute - MINUTE_INTERVAL, minute + MINUTE_INTERVAL)
    

# Function get_CMIPC_data_fileKeys
# Inputs:
#    year       : int 2017-2024              
#    day        : int 1 - 366              
#    hour       : int 0 - 23                           
#    bandNum    : int 1 - 16 
#    min_minute : int 0 - 59        
#    max_minute : int 0 - 59
#
# Note: Scans will be found in the range from min_minute to max_minute inclusive
#
# Returns:
#    dataset obj containing relevant GOES-16 data 
# Example:
#                  YEAR DAY HOUR                 BAND      Start scan      End scan        Midpt b/t start and end time
#                    V   V   V                    V        V               V               V
# Key: ABI-L2-CMIPC/2023/001/01/OR_ABI-L2-CMIPC-M6C14_G16_s20230010101173_e20230010103546_c20230010104092.nc

def get_CMIPC_data_fileKeys(year, day, hour, bandNum, min_minute, max_minute):
    # Set up boto3 client that will access the bucket w/o authentication/creds.
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    # ensure we are only using CMIPC data (Cloud and Moisture Imagery Product for CONUS)
    prefix = 'ABI-L2-CMIPC/'
    
    # convert to yyyy/ddd/hh file prefix used to look up file in bucket
    prefix += f'{year}/{day:03}/{hour:02}'

    # ask bucket for all CMIPC data taken in hour hour on day day in year year
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    except:
        return []
    
    if "Contents" in response:
        fileList = []
        for object in response['Contents']:
            fileKey = object['Key']
                
            if f"C{bandNum:02}" in fileKey:
                # Gets the starting index of the minute field for the scan start time of the file
                startMinIdx = fileKey.index(SCAN_START_INDICATOR) + SCAN_START_INDICATOR_LENGTH + START_MINUTE_OFFSET

                if min_minute <= int(fileKey[startMinIdx:startMinIdx + MINUTE_FIELD_WIDTH]) <= max_minute:
                    print(f'Found file with key: {fileKey}')  # Print the file key
                    fileList.append(fileKey)
        return fileList
    else:
        return []
            
# retrieve_s3_data
# Inputs:
#   fileKey  : string -- key of a specific file in the bucket

def retrieve_s3_data(fileKey):
    # Read the NetCDF data directly into an xarray Dataset
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    s3_object = s3.get_object(Bucket=BUCKET_NAME, Key=fileKey)

    data = s3_object['Body'].read()
    # Open the file and read the data
    nc_dataset = None
    with io.BytesIO(data) as data_io:
        nc_dataset = Dataset('in_memory.nc', mode='r', memory=data_io.read())

    return nc_dataset

# show_CMIPC_image
# Inputs
#   ds : a netCDF4 Dataset containing CMIPC data

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
#
# NOTE: tmpFname MUST REALLY be different than any current filename in your directory
def show_CMIPC_from(year, day, hour, minute, bandNum, idx=0):
    fkeys = get_range_CMIPC_data_fileKeys(year, day, hour, minute, bandNum)

    if idx < len(fkeys):
        ds = retrieve_s3_data(fkeys[idx])

        show_CMPIC_image(ds)
        ds.close()

if __name__ == "__main__":
    show_CMIPC_from(2019, 364, 23, 61, 14) # change dates for testing