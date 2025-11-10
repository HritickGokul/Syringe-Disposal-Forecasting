import requests
import pandas as pd
import os

# Method 1:
# Auto download 1000 latest data(sort by open_dt) by using API.
# Define API URL and Resource ID
API_URL = "https://data.boston.gov/api/3/action/datastore_search"
RESOURCE_ID = "9d7c2214-4709-478a-a2e8-fb2020a5bb94"

# Set request parameters
params = {
    "resource_id": RESOURCE_ID,  # Resource ID of the dataset
    "limit": 1000,  # Limit the number of records to return to 1000
    "sort": "open_dt desc"  # Sort by date in descending order
}

# Define the output file path
output_directory = r"C:\Users\17728\Desktop\ALY6980\WEEK3&4\作业"
output_file = os.path.join(output_directory, "311_latest_data.csv")

# Check if the directory exists, create it if not
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Send the request
response = requests.get(API_URL, params=params)

# Check the request status and process the data
if response.status_code == 200:
    data = response.json()
    records = data["result"]["records"]  # Extract the records

    if records:  # If data is available
        print(f"Number of records retrieved: {len(records)}")  # Print the number of records

        # Convert the data to a DataFrame
        df = pd.DataFrame(records)

        # Save the data as a CSV file
        try:
            df.to_csv(output_file, index=False)
            print(f"Data successfully sorted by the latest date and saved to {output_file}")
        except Exception as e:
            print(f"An error occurred while saving the file: {e}")
    else:
        print("No data retrieved. Please check the API parameters.")
else:
    print(f"Request failed with status code: {response.status_code}")
    print("Error message:", response.text)

# Method 2:
# Auto-download data from each day in 2025 to each day in 2025 by using API.
# If you want to auto-download the dataset in another year, you need to change the resource_id.

# Define API URL and Resource ID
API_URL = "https://data.boston.gov/api/3/action/datastore_search"
RESOURCE_ID = "9d7c2214-4709-478a-a2e8-fb2020a5bb94"

# User-defined date range(Change the date to download)
start_date = "2025-01-01"  # Start date
end_date = "2025-01-02"    # End date

# Define output file path
output_directory = r"C:\Users\17728\Desktop\ALY6980\WEEK3&4\作业"
output_file = os.path.join(output_directory, f"311_latest_data_{start_date}_to_{end_date}.csv")

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

print(f"Downloading data from {start_date} to {end_date}...")

# Set API request parameters to fetch all data (retrieve all matching records in one request)
params = {
    "resource_id": RESOURCE_ID,
    "limit": 50000,  # Fetch as much data as possible to avoid pagination
    "sort": "open_dt desc"  # Sort by date in descending order
}

# Send the request
response = requests.get(API_URL, params=params)

# Check the request status and process the data
if response.status_code == 200:
    data = response.json()
    records = data["result"]["records"]  # Extract records

    # Filter records within the specified date range
    filtered_records = [r for r in records if start_date <= r["open_dt"][:10] <= end_date]

    if filtered_records:
        print(f"Number of records retrieved: {len(filtered_records)}")  # Print the number of records

        # Convert the data to a DataFrame
        df = pd.DataFrame(filtered_records)

        # Save the data as a CSV file
        try:
            df.to_csv(output_file, index=False)
            print(f"Data successfully sorted by the latest date and saved to {output_file}")
        except Exception as e:
            print(f"An error occurred while saving the file: {e}")
    else:
        print("No data found within the specified date range. Please check the start and end dates.")
else:
    print(f"Request failed with status code: {response.status_code}")
    print("Error message:", response.text)
