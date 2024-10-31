import boto3
import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from botocore import UNSIGNED
from botocore.client import Config
from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt

# Initialize the S3 client
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Define the S3 bucket name and prefix
bucket_name = 'noaa-goes16'
prefix = 'ABI-L2-CMIPC/2023/'

# Define the date range for 2023
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
count = 0

# Function to retrieve and manipulate CMIPC data
def retrieve_and_manipulate_cmipc_data():
    current_date = start_date
    while current_date <= end_date:
        month_str = f'{current_date.month:02}'  # Format month as two digits
        day_str = f'{current_date.day:02}'      # Format day as two digits
        print(month_str)
        # print(day_str)
        
        # Construct the prefix for the current month and day
        prefix_day = f'{prefix}0{month_str}/{day_str}/'

        # List the objects in the specified prefix
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix_day)

        # Check if there are any contents
        if 'Contents' in response:
            for obj in response['Contents']:
                file_key = obj['Key']
                print(f'Processing file: {file_key}')  # Print the file key

                # Read the NetCDF data directly into an xarray Dataset
                obj_response = s3.get_object(Bucket=bucket_name, Key=file_key)
                data = obj_response['Body'].read()  # Read the file data
                # Open the file and read the data
                s3.download_file(bucket_name, file_key, Filename=f"my_example")
                # Open the NetCDF data with xarray
                # with xr.open_dataset(xr.backends.NetCDF4DataStore(data)) as ds:
                    # Example manipulation: Print dataset information
                ds = Dataset("my_example", 'r')
                for var in ds.variables:
                    print(var)
                CMI = ds.variables['CMI'][:]
                fig = plt.figure(figsize=(4,4),dpi=200)
                im = plt.imshow(CMI, cmap='Greys_r')
                # cb = fig.colorbar(im, orientation='horizontal')
                # cb.set_ticks([1, 100, 200, 300, 400, 500, 600])
                # cb.set_label('Radiance (W m-2 sr-1 um-1)')
                plt.show()
                # print(ds.get_variables_by_attributes())
                # print(ds)
                # print(ds.__dict__)    # prints metadata

                
                break
        break
        current_date += timedelta(days=1)

retrieve_and_manipulate_cmipc_data()
