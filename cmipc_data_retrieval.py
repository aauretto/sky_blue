import boto3
# import xarray as xr
import numpy as np
from datetime import datetime, timedelta
from botocore import UNSIGNED
from botocore.client import Config

# Initialize the S3 client
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# Define the S3 bucket name and prefix
bucket_name = 'noaa-goes16'
prefix = 'ABI-L2-CMIPC/2023/'

# Define the date range for 2023
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

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
                # # Open the NetCDF data with xarray
                # with xr.open_dataset(xr.backends.NetCDF4DataStore(data)) as ds:
                #     # Example manipulation: Print dataset information
                #     print(ds)
                    
                #     # Example: Calculate mean of a variable (adjust variable name accordingly)
                #     if 'your_variable_name' in ds.variables:
                #         mean_value = ds['your_variable_name'].mean().values
                #         print(f'Mean of your_variable_name: {mean_value}')
                #     else:
                #         print("Variable not found in the dataset.")

        current_date += timedelta(days=1)

retrieve_and_manipulate_cmipc_data()
