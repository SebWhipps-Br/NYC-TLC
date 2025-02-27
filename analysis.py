import pandas as pd
from data_loader import get_from_url, get_from_file

# Change the filename and URL strings as applicable

# Copied URL from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
filename = "yellow_tripdata_2024-01.parquet"

# Set pandas display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)        # Use full width

# Load data (choose one method)
# trips, df = get_from_url(url)  # Uncomment to use URL
trips, df = get_from_file(filename)  # Using local file by default

# Print first few rows
print(df.head())

# Calculate statistics
total_rows = len(df)

# Count rows where store_and_fwd_flag is 'Y'
y_count = len(df[df['store_and_fwd_flag'] == 'Y'])

y_percentage = (y_count / total_rows) * 100

print(f"Total rows: {total_rows}")
print(f"Rows with 'Y': {y_count}")
print(f"Percentage of 'Y' rows: {y_percentage:.2f}%")