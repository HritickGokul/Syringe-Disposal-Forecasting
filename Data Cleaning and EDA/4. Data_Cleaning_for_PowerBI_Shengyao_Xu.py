import pandas as pd
import os
import geopandas as gpd
import requests
import certifi
from shapely.geometry import Point
from shapely import wkt 

# Define your local data directory
data_directory = r"C:\Users\17728\Desktop\ALY6980\WEEK9&10\作业"

# Define file paths for each year
file_paths = {
    "2021": os.path.join(data_directory, "2021.csv"),
    "2022": os.path.join(data_directory, "2022.csv"),
    "2023": os.path.join(data_directory, "2023.csv"),
    "2024": os.path.join(data_directory, "2024.csv"),
    "2025": os.path.join(data_directory, "2025.csv")
}

# Load datasets into a dictionary
dataframes = {}
for year, path in file_paths.items():
    if os.path.exists(path):  # Check if file exists
        try:
            dataframes[year] = pd.read_csv(path, low_memory=False)
            print(f"Successfully loaded {year} data: {dataframes[year].shape[0]} rows, {dataframes[year].shape[1]} columns")
        except Exception as e:
            print(f"Failed to load {year} data: {e}")
    else:
        print(f"File {path} not found. Please check if the file exists.")

# Standardize column names (convert to lowercase and remove spaces)
for year, df in dataframes.items():
    df.columns = df.columns.str.lower().str.strip()

# Merge all years into a single dataset
boston_311_dataset = pd.concat(dataframes.values(), ignore_index=True)

# Display column names
print("\nColumn names in the dataset:")
print(boston_311_dataset.columns.tolist())

# Check missing values
missing_values = boston_311_dataset.isnull().sum()

# Calculate the percentage of missing values
missing_percentage = (missing_values / len(boston_311_dataset)) * 100

# Create a DataFrame for missing values summary
missing_data = pd.DataFrame({
    "Missing Count": missing_values,
    "Missing Percentage (%)": missing_percentage
}).sort_values(by="Missing Count", ascending=False)

# Display missing values summary
print("\nMissing Values Report:")
print(missing_data)

# Remove columns with excessive missing values
columns_to_drop = ["submitted_photo", "closed_photo", "sla_target_dt"]
boston_311_dataset.drop(columns=columns_to_drop, inplace=True, errors="ignore")

# Filter rows where 'reason' is 'Needle Program'
boston_311_needle_program_dataset = boston_311_dataset[boston_311_dataset["reason"] == "Needle Program"]
boston_311_needle_program_dataset

# Define file paths for shapefiles
zip_code_shapefile = r"C:\Users\17728\Desktop\ALY6980\WEEK9&10\作业\zipcodes\ZIPCODES_NT_POLY.shp"
city_council_shapefile = r"C:\Users\17728\Desktop\ALY6980\WEEK9&10\作业\city_council_districts___2023_2032\City_Council_Districts___2023_2032.shp"

# Load shapefiles as GeoDataFrames
zip_code_gdf = gpd.read_file(zip_code_shapefile)
city_council_gdf = gpd.read_file(city_council_shapefile)

# Display column names to find the correct ZIP Code column
print("\nColumns in ZIP Code Shapefile:")
print(zip_code_gdf.columns)

print("\nColumns in City Council Districts Shapefile:")
print(city_council_gdf.columns)

# Convert boston_311_needle_program_dataset to a GeoDataFrame using latitude and longitude
boston_311_needle_program_dataset = boston_311_needle_program_dataset.copy()
boston_311_needle_program_dataset["geometry"] = boston_311_needle_program_dataset.apply(
    lambda row: Point(row["longitude"], row["latitude"]), axis=1
)

geo_boston_311_needle_program = gpd.GeoDataFrame(
    boston_311_needle_program_dataset, geometry="geometry", crs="EPSG:4326"
)

# Ensure CRS consistency with shapefiles
geo_boston_311_needle_program = geo_boston_311_needle_program.to_crs(zip_code_gdf.crs)
city_council_gdf = city_council_gdf.to_crs(zip_code_gdf.crs)

# Perform spatial join to correct location_zipcode
geo_boston_311_needle_program = gpd.sjoin(
    geo_boston_311_needle_program, zip_code_gdf[["POSTCODE", "geometry"]], how="left", predicate="intersects"
)
geo_boston_311_needle_program["location_zipcode"] = geo_boston_311_needle_program["POSTCODE"]

# Drop index_right before second spatial join to avoid conflicts
geo_boston_311_needle_program.drop(columns=["index_right"], inplace=True, errors="ignore")

# Perform spatial join to correct city_council_district
geo_boston_311_needle_program = gpd.sjoin(
    geo_boston_311_needle_program, city_council_gdf[["DISTRICT", "geometry"]], how="left", predicate="intersects"
)
geo_boston_311_needle_program["city_council_district"] = geo_boston_311_needle_program["DISTRICT"]

# Drop unnecessary columns (spatial join artifacts)
columns_to_remove = ["POSTCODE", "DISTRICT", "index_right", "geometry"]
boston_311_needle_program_dataset = geo_boston_311_needle_program.drop(columns=columns_to_remove, errors="ignore")

# Drop unnecessary columns if not needed
columns_to_remove = ["geom_4326"]  
boston_311_needle_program_dataset = boston_311_needle_program_dataset.drop(columns=columns_to_remove, errors="ignore")

# Display the rows to confirm corrections
boston_311_needle_program_dataset

# Define columns that must not have missing values
columns_to_check = ["city_council_district", "location_zipcode", "latitude", "longitude", "neighborhood"]

# Drop rows where any of these columns are missing
boston_311_needle_program_dataset.dropna(subset=columns_to_check, inplace=True)

# Display the updated missing values report
missing_values_final = boston_311_needle_program_dataset.isnull().sum()
missing_percentage_final = (missing_values_final / len(boston_311_needle_program_dataset)) * 100

# Create a DataFrame for missing values summary
missing_data_final = pd.DataFrame({
    "Missing Count": missing_values_final,
    "Missing Percentage (%)": missing_percentage_final
}).sort_values(by="Missing Count", ascending=False)

# Display missing values report after dropping rows
print("\nFinal Missing Values Report after Removing Incomplete Rows:")
print(missing_data_final)

# ArcGIS API endpoint (Layer ID = 111)
arcgis_url = "https://gis.bostonplans.org/hosting/rest/services/FY23_Parcels_with_Planning_and_Zoning_Info/FeatureServer/111/query"

# Pagination parameters
batch_size = 2000  # Maximum records per request
offset = 0
all_features = []  # List to store all fetched data

while True:
    print(f"Fetching data, offset: {offset}...")

    params = {
        "where": "1=1",  # Retrieve all records
        "outFields": "*",  # Retrieve all fields
        "returnGeometry": "true",  # Include geographic coordinates
        "f": "geojson",  # Request response in GeoJSON format
        "resultOffset": offset,  # Pagination offset
        "resultRecordCount": batch_size  # Number of records per request (max 2000)
    }

    response = requests.get(arcgis_url, params=params, verify=certifi.where())

    if response.status_code == 200:
        data = response.json()
        if "features" in data and len(data["features"]) > 0:
            all_features.extend(data["features"])  # Append records to the list
            offset += batch_size  # Move to the next batch
        else:
            print("All data has been successfully downloaded!")
            break
    else:
        print("API request failed:", response.status_code, response.text)
        break

# Convert the collected data into a GeoDataFrame
property_gdf = gpd.GeoDataFrame.from_features(all_features)

# **Ensure CRS (Coordinate Reference System) is correctly set**
if property_gdf.crs is None:
    property_gdf.set_crs("EPSG:4326", inplace=True)  # Set CRS to WGS 84

print(f"Data retrieval complete! Total records: {len(property_gdf)}")

# **Save the full dataset**
property_gdf.to_file("Boston_Parcels_Full.geojson", driver="GeoJSON")  # Save as GeoJSON
property_gdf.drop(columns=['geometry']).to_csv("Boston_Parcels_Full.csv", index=False)  # Save as CSV

print("Data saved successfully!")

# Ensure CRS consistency between datasets
property_gdf = property_gdf.to_crs(geo_boston_311_needle_program.crs)

# Drop 'index_right' if it already exists in either dataset
geo_boston_311_needle_program.drop(columns=["index_right"], inplace=True, errors="ignore")
property_gdf.drop(columns=["index_right"], inplace=True, errors="ignore")

# Perform spatial join to assign owner names based on geographic location
geo_boston_311_needle_program = gpd.sjoin(
    geo_boston_311_needle_program, property_gdf[["OWNER", "geometry"]], how="left", predicate="intersects"
)

# Assign owner_name from spatial join result
geo_boston_311_needle_program["owner_name"] = geo_boston_311_needle_program["OWNER"]

# Drop unnecessary columns (spatial join artifacts)
columns_to_remove = ["index_right", "OWNER", "geometry"]
boston_311_needle_program_dataset = geo_boston_311_needle_program.drop(columns=columns_to_remove, errors="ignore")

# Display a sample of the updated dataset
print("\nUpdated dataset with owner_name:")
print(boston_311_needle_program_dataset.head())

# Define file paths
public_schools_path = r"C:\Users\17728\Desktop\ALY6980\WEEK9&10\作业\public_schools.csv"
non_public_schools_path = r"C:\Users\17728\Desktop\ALY6980\WEEK9&10\作业\non_public_schools.csv"
open_space_path = r"C:\Users\17728\Desktop\ALY6980\WEEK9&10\作业\open_space.csv"

# Load datasets
public_schools = pd.read_csv(public_schools_path)
non_public_schools = pd.read_csv(non_public_schools_path)
open_space = pd.read_csv(open_space_path)

# Print column names to identify potential issues
print("\nColumns in public_schools dataset:", public_schools.columns.tolist())
print("\nColumns in non_public_schools dataset:", non_public_schools.columns.tolist())
print("\nColumns in open_space dataset:", open_space.columns.tolist())

# Load datasets
public_schools = pd.read_csv(public_schools_path)
non_public_schools = pd.read_csv(non_public_schools_path)
open_space = pd.read_csv(open_space_path)

# Convert public and non-public schools to GeoDataFrames using POINT_X (longitude) & POINT_Y (latitude)
public_schools_gdf = gpd.GeoDataFrame(
    public_schools, geometry=gpd.points_from_xy(public_schools["POINT_X"], public_schools["POINT_Y"]), crs="EPSG:4326"
)

non_public_schools_gdf = gpd.GeoDataFrame(
    non_public_schools, geometry=gpd.points_from_xy(non_public_schools["POINT_X"], non_public_schools["POINT_Y"]), crs="EPSG:4326"
)

# Convert 'shape_wkt' to geometry using shapely.wkt.loads()
open_space["shape_wkt"] = open_space["shape_wkt"].astype(str)
open_space = open_space.dropna(subset=["shape_wkt"])
open_space["geometry"] = open_space["shape_wkt"].apply(lambda x: wkt.loads(x) if isinstance(x, str) and x.startswith(("POINT", "POLYGON", "MULTIPOLYGON")) else None)
open_space = open_space.dropna(subset=["geometry"])
open_space_gdf = gpd.GeoDataFrame(open_space, geometry="geometry", crs="EPSG:4326")

# Ensure CRS consistency
geo_boston_311_needle_program = geo_boston_311_needle_program.to_crs(public_schools_gdf.crs)

# **Convert all datasets to a projected CRS (EPSG:26986 for Boston)**
projected_crs = "EPSG:26986"

geo_boston_311_needle_program = geo_boston_311_needle_program.to_crs(projected_crs)
public_schools_gdf = public_schools_gdf.to_crs(projected_crs)
non_public_schools_gdf = non_public_schools_gdf.to_crs(projected_crs)
open_space_gdf = open_space_gdf.to_crs(projected_crs)

# Function to check if a point is within 50 meters of any features in a dataset
def is_nearby(needle_points, reference_gdf, distance_threshold=50):
    return needle_points.geometry.apply(lambda x: reference_gdf.distance(x).min() <= distance_threshold)

# Create new boolean columns for proximity analysis
geo_boston_311_needle_program["near_public_school"] = is_nearby(geo_boston_311_needle_program, public_schools_gdf)
geo_boston_311_needle_program["near_non_public_school"] = is_nearby(geo_boston_311_needle_program, non_public_schools_gdf)
geo_boston_311_needle_program["near_open_space"] = is_nearby(geo_boston_311_needle_program, open_space_gdf)

# Convert boolean columns to integers (0 = No, 1 = Yes) for Power BI
geo_boston_311_needle_program["near_public_school"] = geo_boston_311_needle_program["near_public_school"].astype(int)
geo_boston_311_needle_program["near_non_public_school"] = geo_boston_311_needle_program["near_non_public_school"].astype(int)
geo_boston_311_needle_program["near_open_space"] = geo_boston_311_needle_program["near_open_space"].astype(int)

# Convert back to geographic CRS (EPSG:4326) for Power BI mapping
geo_boston_311_needle_program = geo_boston_311_needle_program.to_crs("EPSG:4326")

# Drop unnecessary spatial columns
boston_311_needle_program_dataset = geo_boston_311_needle_program.drop(columns=["geometry"], errors="ignore")
print(boston_311_needle_program_dataset.head())

# Display all column names
print("\nColumns in boston_311_needle_program_dataset:")
print(boston_311_needle_program_dataset.columns.tolist())

# Check for missing values
missing_values = boston_311_needle_program_dataset.isnull().sum()

# Calculate the percentage of missing values
missing_percentage = (missing_values / len(boston_311_needle_program_dataset)) * 100

# Create a DataFrame for missing values summary
missing_data_report = pd.DataFrame({
    "Missing Count": missing_values,
    "Missing Percentage (%)": missing_percentage
}).sort_values(by="Missing Count", ascending=False)

# Display missing values summary
print("\nMissing Values Report:")
print(missing_data_report)

# Define the required columns for Power BI
required_columns = [
    "case_enquiry_id", "open_dt", "closed_dt", "on_time", "case_status", 
    "closure_reason", "case_title", "subject", "reason", "type", "queue", 
    "location", "location_street_name", "location_zipcode", "latitude", 
    "longitude", "city_council_district", "neighborhood", "fire_district", 
    "pwd_district", "police_district", "precinct", "ward", 
    "neighborhood_services_district", "source", "owner_name",
    "near_public_school", "near_non_public_school", "near_open_space"
]

# Ensure the dataset only contains relevant columns
boston_311_needle_program_dataset_cleaned = boston_311_needle_program_dataset[required_columns]
boston_311_needle_program_dataset_cleaned

# Define the file path for the cleaned dataset
cleaned_file_path = os.path.join(data_directory, "boston_311_needle_program_dataset_cleaned.csv")

# Save the cleaned dataset as CSV
boston_311_needle_program_dataset_cleaned.to_csv(cleaned_file_path, index=False, encoding="utf-8")

print(f"\nCleaned dataset successfully saved to: {cleaned_file_path}")

# Set data directory
data_directory = r"C:\Users\17728\Desktop\ALY6980\WEEK9&10\作业"

# File paths
files = {
    "boston_311": "boston_311_needle_program_dataset_cleaned.csv",
    "public_schools": "public_schools.csv",
    "non_public_schools": "non_public_schools.csv",
    "open_space": "open_space.csv"
}

# Load datasets
dfs = {}
for key, file in files.items():
    file_path = os.path.join(data_directory, file)
    df = pd.read_csv(file_path, low_memory=False)
    dfs[key] = df
    print(f"{key} dataset column names and data types:\n{df.dtypes}\n")

# Rename latitude and longitude columns for consistency
rename_mappings = {
    "public_schools": {"POINT_Y": "latitude", "POINT_X": "longitude"},
    "non_public_schools": {"POINT_Y": "latitude", "POINT_X": "longitude"},
}

for key, mapping in rename_mappings.items():
    if key in dfs:
        dfs[key] = dfs[key].rename(columns=mapping)

# Process boston_311 dataset (Extract year and month from open_dt)
boston_311_cleaned = dfs["boston_311"][["latitude", "longitude", "open_dt"]].copy()
boston_311_cleaned["category"] = "Needle Program"

# Convert 'open_dt' to datetime format
boston_311_cleaned["open_dt"] = pd.to_datetime(boston_311_cleaned["open_dt"], errors="coerce")

# Extract year and month
boston_311_cleaned["year"] = boston_311_cleaned["open_dt"].dt.year
boston_311_cleaned["month"] = boston_311_cleaned["open_dt"].dt.month

# Drop original open_dt column
boston_311_cleaned.drop(columns=["open_dt"], inplace=True)

# Generate a DataFrame for all months from Jan 2021 to Feb 2025
months_list = pd.date_range(start="2021-01-01", end="2025-02-01", freq="MS")
month_year_df = pd.DataFrame({"year": months_list.year, "month": months_list.month})

# Process public_schools and non_public_schools (Expand across all months)
public_schools_cleaned = dfs["public_schools"][["latitude", "longitude"]].copy()
public_schools_cleaned["category"] = "Public School"

non_public_schools_cleaned = dfs["non_public_schools"][["latitude", "longitude"]].copy()
non_public_schools_cleaned["category"] = "Non-Public School"

open_space_df = dfs["open_space"][["shape_wkt"]].copy()
open_space_df["category"] = "Open Space"

# Convert WKT to Shapely geometry and compute centroids
open_space_df["geometry"] = open_space_df["shape_wkt"].apply(lambda x: wkt.loads(str(x)) if isinstance(x, str) and x.startswith(("POLYGON", "MULTIPOLYGON")) else None)
open_space_df.dropna(subset=["geometry"], inplace=True)
open_space_gdf = gpd.GeoDataFrame(open_space_df, geometry="geometry", crs="EPSG:4326")

# Reproject before centroid extraction
open_space_gdf = open_space_gdf.to_crs(epsg=3857)
open_space_gdf["geometry"] = open_space_gdf.geometry.centroid
open_space_gdf = open_space_gdf.to_crs(epsg=4326)

# Extract lat/lon
open_space_cleaned = open_space_gdf[["geometry"]].copy()
open_space_cleaned["latitude"] = open_space_cleaned.geometry.y
open_space_cleaned["longitude"] = open_space_cleaned.geometry.x
open_space_cleaned["category"] = "Open Space"
open_space_cleaned.drop(columns=["geometry"], inplace=True)

# Replicate non-changing datasets for every month from 2021-01 to 2025-02
def replicate_across_months(df, category):
    df["key"] = 1  # Create a key for cross join
    month_year_df["key"] = 1
    expanded_df = df.merge(month_year_df, on="key").drop(columns=["key"])
    expanded_df["category"] = category
    return expanded_df

public_schools_expanded = replicate_across_months(public_schools_cleaned, "Public School")
non_public_schools_expanded = replicate_across_months(non_public_schools_cleaned, "Non-Public School")
open_space_expanded = replicate_across_months(open_space_cleaned, "Open Space")

# Merge all datasets into one final dataset
final_dataset = pd.concat([boston_311_cleaned, public_schools_expanded, non_public_schools_expanded, open_space_expanded], ignore_index=True)

# Fill missing values
final_dataset.fillna({"longitude": 0, "latitude": 0, "category": "Unknown"}, inplace=True)

# Define U.S. boundary latitude and longitude ranges
us_latitude_range = (24.396308, 49.384358)  # Min, Max Latitude
us_longitude_range = (-125.000000, -66.934570)  # Min, Max Longitude

# Filter dataset to keep only rows within the U.S.
final_dataset = final_dataset[
    (final_dataset["latitude"].between(*us_latitude_range)) &
    (final_dataset["longitude"].between(*us_longitude_range))
]

# Save the final dataset
output_file = os.path.join(data_directory, "final_combined_dataset.csv")
final_dataset.to_csv(output_file, index=False, encoding="utf-8")

print(f"Final dataset saved to {output_file}, ready for import into Power BI.")
