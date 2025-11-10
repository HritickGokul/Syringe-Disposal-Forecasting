import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import certifi
import contextily as ctx
import matplotlib.patches as mpatches
from shapely.geometry import Point
from sklearn.impute import KNNImputer
from statsmodels.tsa.statespace.sarimax import SARIMAX
from shapely import wkt

boston_311_2024_df = pd.read_csv('2024.csv')

boston_311_2024_df.head()

# Get the number of rows and columns
rows, columns = boston_311_2024_df.shape

# Get the column names
column_names = boston_311_2024_df.columns.tolist()

# Print the result
print(f"The dataset has {rows} rows and {columns} columns.")
print("The column names are:")
print(column_names)

boston_311_2024_df.info()

# Let's summarize the data to see the distribution of data
print(boston_311_2024_df.describe())   

# Calculate the number of unique values
unique_values = boston_311_2024_df.nunique()

# Print the result
print(unique_values)

# Filter columns with fewer than 20 unique values
columns_with_few_unique_values = unique_values[unique_values < 50]

# Print the result
print("Columns with fewer than 20 unique values and their unique values:")
for column in columns_with_few_unique_values.index:
    print(f"\nColumn: {column}")
    print(boston_311_2024_df[column].unique())

# Check the missing values
print("Missing values count in each column:")
print(boston_311_2024_df.isnull().sum())

# Check the NA values (including NaN and "NA" strings)
na_values = (boston_311_2024_df == 'NA').sum()

# Print the result in the same format as isnull().sum()
print("NA values count in each column:")
print(na_values)

# Drop the useless columns
boston_311_2024_df.drop(columns=['submitted_photo', 'closed_photo'], inplace=True)

# Print the updated shape of the dataframe
print(f"Updated dataset shape: {boston_311_2024_df.shape}")

# Extract rows where latitude or longitude is missing
mis_la_lo_df = boston_311_2024_df[boston_311_2024_df['latitude'].isnull() | boston_311_2024_df['longitude'].isnull()]

# Display the first five rows of the missing data
print(mis_la_lo_df.head())

# Count rows where either location_street_name or location_zipcode is not null
valid_location_info_count = mis_la_lo_df[
    mis_la_lo_df['location_street_name'].notnull() | mis_la_lo_df['location_zipcode'].notnull()
].shape[0]

# Print the result
print(f"Number of rows with either location_street_name or location_zipcode: {valid_location_info_count}")

# Print the dataset shape
print(f"Updated dataset shape before removal: {boston_311_2024_df.shape}")

# Remove rows where latitude or longitude is missing
boston_311_2024_df = boston_311_2024_df.dropna(subset=['latitude', 'longitude'])

# Print the updated dataset shape
print(f"Updated dataset shape after removal: {boston_311_2024_df.shape}")

# Check the missing values again
print("Missing values count in each column:")
print(boston_311_2024_df.isnull().sum())

# Import the City Council dataset and check the usability
# Load shapefile in the City Council Districts File
shapefile_path = "C:/Users/17728/Desktop/ALY6980/WEEK7&8/作业/city_council_districts___2023_2032/City_Council_Districts___2023_2032.shp"
city_council_gdf = gpd.read_file(shapefile_path)

# Check the five rows in the dataset
print(city_council_gdf.head())

# Check the columns
print(city_council_gdf.columns)

# Convert the DataFrame to a GeoDataFrame using latitude and longitude
boston_311_2024_df["geometry"] = boston_311_2024_df.apply(
    lambda row: Point(row["longitude"], row["latitude"]), axis=1
)
geo_boston_311_2024 = gpd.GeoDataFrame(
    boston_311_2024_df, geometry="geometry", crs="EPSG:4326"
)

# Ensure both GeoDataFrames have the same CRS
geo_boston_311_2024 = geo_boston_311_2024.to_crs(city_council_gdf.crs)

# Perform spatial join to assign city council districts
geo_boston_311_2024 = gpd.sjoin(
    geo_boston_311_2024, city_council_gdf[["DISTRICT", "geometry"]], how="left", predicate="intersects"
)

# Fix the city_council_district column
geo_boston_311_2024["city_council_district"] = geo_boston_311_2024["DISTRICT"]

# Drop unnecessary columns
cleaned_boston_311_2024_df = geo_boston_311_2024.drop(columns=["DISTRICT", "index_right"])
cleaned_boston_311_2024_df.head()

# Check the missing values again
print("Missing values count in each column:")
print(cleaned_boston_311_2024_df.isnull().sum())

# Filter rows where city_council_district is missing
d_boston_311_2024_df = cleaned_boston_311_2024_df[cleaned_boston_311_2024_df['city_council_district'].isnull()]

# Display the first five rows of the filtered DataFrame
print(d_boston_311_2024_df)

# Export to a CSV file and check the dataset
d_boston_311_2024_df.to_csv("missing_city_council_districts.csv", index=False)

# Create GeoDataFrame
missing_gdf = gpd.GeoDataFrame(
    d_boston_311_2024_df,
    geometry=[Point(xy) for xy in zip(d_boston_311_2024_df.longitude, d_boston_311_2024_df.latitude)],
    crs="EPSG:4326"
)

# Check if those points are in the area of shapefile (use union_all instead of unary_union)
inside_check = missing_gdf.within(city_council_gdf.geometry.union_all())

print("Number of missing points inside the city council boundary:", inside_check.sum())

# Fill missing values in city_council_district using the original dataset
cleaned_boston_311_2024_df["city_council_district"] = cleaned_boston_311_2024_df["city_council_district"].fillna(
    boston_311_2024_df["city_council_district"]
)

# Check the missing values again
print("Missing values count in each column:")
print(cleaned_boston_311_2024_df.isnull().sum())

# Load the ZIP Code shapefile
zip_code_shapefile = "C:/Users/17728/Desktop/ALY6980/WEEK7&8/作业/zipcodes/ZIPCODES_NT_POLY.shp"
zip_gdf = gpd.read_file(zip_code_shapefile)

# Check the five rows in the dataset
print(zip_gdf.head())

# Check the columns
print(zip_gdf.columns)

# Convert the main DataFrame to GeoDataFrame
cleaned_boston_311_2024_df["geometry"] = cleaned_boston_311_2024_df.apply(
    lambda row: Point(row["longitude"], row["latitude"]), axis=1
)
boston_gdf = gpd.GeoDataFrame(cleaned_boston_311_2024_df, geometry="geometry", crs="EPSG:4326")

# Ensure the CRS matches
if zip_gdf.crs != boston_gdf.crs:
    boston_gdf = boston_gdf.to_crs(zip_gdf.crs)

# Perform spatial join
boston_gdf = gpd.sjoin(boston_gdf, zip_gdf, how="left", predicate="intersects")

# Fill missing ZIP Codes
boston_gdf["location_zipcode"] = boston_gdf["POSTCODE"].combine_first(
    cleaned_boston_311_2024_df["location_zipcode"]
)

# Update the original DataFrame and save
cleaned_boston_311_2024_df["location_zipcode"] = boston_gdf["location_zipcode"]
cleaned_boston_311_2024_df.head()

# Check the missing values again
print("Missing values count in each column:")
print(cleaned_boston_311_2024_df.isnull().sum())

# Filter rows where location_zipcode is missing
z_boston_311_2024_df = cleaned_boston_311_2024_df[cleaned_boston_311_2024_df['location_zipcode'].isnull()]

# Display the first five rows of the filtered DataFrame
print(z_boston_311_2024_df)

# Export to a CSV file and check the dataset
z_boston_311_2024_df.to_csv("missing_location_zipcode.csv", index=False)

# Create GeoDataFrame for rows with missing location_zipcode
missing_zip_gdf = gpd.GeoDataFrame(
    z_boston_311_2024_df,
    geometry=[Point(xy) for xy in zip(z_boston_311_2024_df.longitude, z_boston_311_2024_df.latitude)],
    crs="EPSG:4326"
)

# Check if those points are within the ZIP Code shapefile area (use union_all instead of unary_union)
zip_boundary_check = missing_zip_gdf.within(zip_gdf.geometry.union_all())

# Output the results
print("Number of missing points inside the ZIP Code boundary:", zip_boundary_check.sum())

# Remove rows where location_zipcode is missing
cleaned_boston_311_2024_df = cleaned_boston_311_2024_df.dropna(subset=["location_zipcode"])

# Check the missing values again
print("Missing values count in each column:")
print(cleaned_boston_311_2024_df.isnull().sum())

# Check the unique value of reason and case title.
print(cleaned_boston_311_2024_df["reason"].unique())
print(cleaned_boston_311_2024_df["case_title"].unique())

# Filter data where the "reason" column is "Needle Program"
syringe_request_df = cleaned_boston_311_2024_df[
    cleaned_boston_311_2024_df["reason"] == "Needle Program"
]
print(syringe_request_df.shape)

# Check the missing values again
print("Missing values count in each column:")
print(syringe_request_df.isnull().sum())

# Group by city council district and count
district_summary = syringe_request_df["city_council_district"].value_counts().reset_index()
district_summary.columns = ["City Council District", "Count"]  # Rename columns

# Convert city council district to integers where possible
district_summary["City Council District"] = pd.to_numeric(district_summary["City Council District"], errors="coerce").fillna(0).astype(int)

# Calculate the proportion of requests for each district
total_requests = district_summary["Count"].sum()
district_summary["Proportion (%)"] = (district_summary["Count"] / total_requests * 100).round(2)

# Print the summary table
print(district_summary)

# Set the proportion threshold (e.g., labels not shown for parts with less than 0.7% proportion)
threshold = 0.7

# Create the pie chart
plt.figure(figsize=(12, 8))  # Adjusted size for legend
wedges, texts, autotexts = plt.pie(
    district_summary["Count"],
    labels=[
        f"{label}" if proportion >= threshold else ""  # Show labels only for proportions >= threshold
        for label, proportion in zip(district_summary["City Council District"], district_summary["Proportion (%)"])
    ],
    autopct=lambda p: f"{p:.1f}%" if p >= threshold else "",  # Show percentages only for proportions >= threshold
    startangle=90,
    colors=plt.cm.Paired.colors,
)

# Add a legend
plt.legend(
    labels=[f"District {label}" for label in district_summary["City Council District"]],
    title="City Council Districts",
    loc="center left",
    bbox_to_anchor=(1, 0.5),  # Position legend to the right of the chart
)

# Adjust font sizes
for text in texts:
    text.set_size(10)
for autotext in autotexts:
    autotext.set_size(8)

# Add title
plt.title("Syringe 311 Requests, By City Council District\nFrom January 01, 2024 to December 31, 2024", pad=20)

# Ensure proper layout and aspect ratio
plt.axis("equal")
plt.tight_layout()
plt.show()

# Ensure 'open_dt' is a string type 
syringe_request_df = syringe_request_df.copy()  # Avoid modifying original DataFrame

# Convert 'open_dt' to datetime format
syringe_request_df['open_dt'] = pd.to_datetime(syringe_request_df['open_dt'], errors='coerce')

# Drop rows with invalid dates (NaT values)
syringe_request_df = syringe_request_df.dropna(subset=['open_dt'])

# Aggregate the number of requests by month
syringe_request_df['month'] = syringe_request_df['open_dt'].dt.to_period('M')
monthly_requests = syringe_request_df.groupby('month').size().reset_index(name='count')

# Convert 'month' period to datetime for plotting
monthly_requests['month'] = monthly_requests['month'].astype(str)
monthly_requests['month'] = pd.to_datetime(monthly_requests['month'])

# Plot the time series line chart
plt.figure(figsize=(12, 6))
plt.plot(monthly_requests['month'], monthly_requests['count'], marker='o', linestyle='-')
plt.xlabel('Month')
plt.ylabel('Syringe Requests Count')
plt.title('Monthly Trend of Syringe Requests')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Build and fit SARIMA model
sarima_model = SARIMAX(
    monthly_requests['count'],
    order=(0, 1, 0),  # Simplified ARIMA parameters
    seasonal_order=(0, 1, 0, 12),  # Seasonality: yearly pattern
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_result = sarima_model.fit(method='powell', disp=False)

# Forecast for the next 3 months
forecast_steps = 3
forecast = sarima_result.get_forecast(steps=forecast_steps).predicted_mean

# Create future date range for plotting
future_dates = pd.date_range(
    monthly_requests['month'].iloc[-1], periods=forecast_steps + 1, freq='ME'
)[1:]

# Plot the forecast along with actual data
plt.figure(figsize=(12, 6))
plt.plot(monthly_requests['month'], monthly_requests['count'], label='Actual')
plt.plot(future_dates, forecast, linestyle='--', marker='o', label='Forecast', color='orange')
plt.xlabel('Month')
plt.ylabel('Syringe Requests Count')
plt.title('Syringe Requests Forecast for Next 3 Months')
plt.legend()
plt.grid(True)
plt.show()

# Print forecast values
print("Forecasted counts for the next 3 months:")
print(pd.DataFrame({'Month': future_dates, 'Forecasted Count': forecast}))

# Summarize requests by city council district
district_summary = syringe_request_df["city_council_district"].value_counts().reset_index()
district_summary.columns = ["City Council District", "Count"]

# Create a deep red to light red gradient palette
gradient_palette = sns.light_palette("red", n_colors=len(district_summary), reverse=True)

# Plot the bar chart with deep red to light red gradient
plt.figure(figsize=(10, 6))
sns.barplot(
    x='City Council District',
    y='Count',
    data=district_summary,
    palette=gradient_palette,  # Apply the gradient palette
    hue='City Council District',  # Set hue to match x variable
    dodge=False
)

# Remove the legend since `hue` duplicates the x variable
plt.legend([], [], frameon=False)

plt.xlabel('City Council District')
plt.ylabel('Syringe Requests Count')
plt.title('Syringe Requests by City Council District')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

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

# Ensure CRS is the same for both datasets
syringe_request_df = syringe_request_df.to_crs("EPSG:4326")
property_gdf = property_gdf.to_crs("EPSG:4326")

# Select relevant fields from property data
property_gdf = property_gdf[['MAP_PAR_ID', 'OWNER', 'Owner_Name', 'YR_BUILT', 
                             'Shape__Area', 'Height_Requirement', 'geometry']]

# Perform spatial join (matches 311 requests to the parcel it falls within)
syringe_request_df = gpd.sjoin(syringe_request_df, property_gdf, how="left", predicate="intersects")

# Rename columns for clarity
syringe_request_df.rename(columns={
    'MAP_PAR_ID': 'Parcel_ID',
    'OWNER': 'Property_Owner',
    'Owner_Name': 'Owner_Name',
    'YR_BUILT': 'Year_Built',
    'Shape__Area': 'Parcel_Area',
    'Height_Requirement': 'Max_Height'
}, inplace=True)

# Save the enriched 311 dataset
syringe_request_df.to_file("311_2024_with_property_data.geojson", driver="GeoJSON")
syringe_request_df.drop(columns=['geometry']).to_csv("311_2024_with_property_data.csv", index=False)

print("311 data successfully matched with property data!")

# Find the top parcels with the most syringe requests**
top_parcels = syringe_request_df['Parcel_ID'].value_counts().head(10)
print("Top 10 parcels with the most syringe requests:")
print(top_parcels)

# Create the bar chart
plt.figure(figsize=(12, 6))
plt.bar(top_parcels.index.astype(str), top_parcels.values, color='royalblue')

# Add labels and title
plt.xlabel('Parcel ID')
plt.ylabel('Number of Syringe Requests')
plt.title('Top 10 Parcels with the Most Syringe Requests')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

# Show the plot
plt.show()

# Find the top property owners with the most syringe requests
top_owners = syringe_request_df['Property_Owner'].value_counts().head(10)
print("\nTop 10 property owners with the most syringe requests:")
print(top_owners)

# Create the bar chart
plt.figure(figsize=(12, 6))
plt.bar(top_owners.index, top_owners.values, color='royalblue')

# Add labels and title
plt.xlabel('Property Owners')
plt.ylabel('Number of Syringe Requests')
plt.title('Top 10 Property Owners with the Most Syringe Requests')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

# Show the plot
plt.show()

# Count the number of requests for buildings before and after 1950
old_buildings = syringe_request_df[syringe_request_df['Year_Built'] < 1950].shape[0]
total_requests = syringe_request_df.shape[0]
new_buildings = total_requests - old_buildings

print(f"Percentage of syringe requests at buildings built before 1950: {old_buildings / total_requests:.2%}")
print(f"Percentage of syringe requests at buildings built after 1950: {new_buildings / total_requests:.2%}")

# Create histogram with density curve
plt.figure(figsize=(12, 6))

# Histogram
sns.histplot(syringe_request_df['Year_Built'], bins=30, kde=True, edgecolor='black', color='royalblue')

# Labels and title
plt.title("Distribution of Syringe Requests by Building Age", fontsize=14)
plt.xlabel("Year Built", fontsize=12)
plt.ylabel("Number of Requests", fontsize=12)

# Show plot
plt.show()

# Load CSV files
public_schools_df = pd.read_csv("public_schools.csv")
non_public_schools_df = pd.read_csv("non_public_schools.csv")
open_space_df = pd.read_csv("open_space.csv")

# Convert public & non-public school data into GeoDataFrames
public_schools_gdf = gpd.GeoDataFrame(
    public_schools_df, 
    geometry=gpd.points_from_xy(public_schools_df['POINT_X'], public_schools_df['POINT_Y']),
    crs="EPSG:4326"
)

non_public_schools_gdf = gpd.GeoDataFrame(
    non_public_schools_df, 
    geometry=gpd.points_from_xy(non_public_schools_df['POINT_X'], non_public_schools_df['POINT_Y']),
    crs="EPSG:4326"
)

# Process open space data (convert WKT to Polygon)
open_space_df.dropna(subset=['shape_wkt'], inplace=True)
open_space_df['geometry'] = open_space_df['shape_wkt'].apply(wkt.loads)
open_space_gdf = gpd.GeoDataFrame(open_space_df, geometry='geometry', crs="EPSG:4326")

# Convert 311 syringe request data into a GeoDataFrame
syringe_request_gdf = gpd.GeoDataFrame(
    syringe_request_df,
    geometry=gpd.points_from_xy(syringe_request_df['longitude'], syringe_request_df['latitude']),
    crs="EPSG:4326"
).drop(columns=['index_right'], errors='ignore')  # Ignore `index_right` if it exists

# Perform spatial joins
syringe_request_gdf = gpd.sjoin(syringe_request_gdf, public_schools_gdf, how="left", predicate="intersects") \
    .rename(columns={'SCH_NAME': 'Nearby_Public_School'}).drop(columns=['index_right'], errors='ignore')

syringe_request_gdf = gpd.sjoin(syringe_request_gdf, non_public_schools_gdf, how="left", predicate="intersects") \
    .rename(columns={'NAME': 'Nearby_NonPublic_School'}).drop(columns=['index_right'], errors='ignore')

syringe_request_gdf = gpd.sjoin(syringe_request_gdf, open_space_gdf, how="left", predicate="intersects") \
    .rename(columns={'SITE_NAME': 'Nearby_Open_Space'}).drop(columns=['index_right'], errors='ignore')

# Calculate KPI (Key Performance Indicators)
requests = [
    syringe_request_gdf['Nearby_Public_School'].notna().sum(),
    syringe_request_gdf['Nearby_NonPublic_School'].notna().sum(),
    syringe_request_gdf['Nearby_Open_Space'].notna().sum(),
]

# Plot bar chart
labels = ["Public Schools", "Non-Public Schools", "Open Spaces"]

fig, ax = plt.subplots(figsize=(8, 6)) 

# Bar chart
bars = ax.bar(labels, requests, color=['blue', 'orange', 'green'])
ax.set_title("Syringe Requests Near Different Locations")
ax.set_ylabel("Number of Requests")
ax.set_xlabel("Location Type")

# Move numerical labels downward to avoid overlapping the bar line
dy = -0  # Adjust downward offset
for bar, value in zip(bars, requests):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + dy, str(value), 
            ha='center', fontsize=12, color='black')

plt.tight_layout()
plt.show()

# Print results
print(f"Syringe Request Results:")
print(f"Syringe requests near public schools: {requests[0]}")
print(f"Syringe requests near non-public schools: {requests[1]}")
print(f"Syringe requests near open spaces (parks/playgrounds): {requests[2]}")

# Convert CRS to Web Mercator (EPSG:3857) for correct map overlay
syringe_request_gdf = syringe_request_gdf.to_crs(epsg=3857)
public_schools_gdf = public_schools_gdf.to_crs(epsg=3857)
non_public_schools_gdf = non_public_schools_gdf.to_crs(epsg=3857)
open_space_gdf = open_space_gdf.to_crs(epsg=3857)

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 10))

# Plot syringe requests (Red)
syringe_request_gdf.plot(ax=ax, color='red', markersize=5, alpha=0.6, label="Syringe Requests")

# Plot public schools (Blue)
public_schools_gdf.plot(ax=ax, color='blue', markersize=30, alpha=0.5, label="Public Schools")

# Plot non-public schools (Green)
non_public_schools_gdf.plot(ax=ax, color='green', markersize=30, alpha=0.5, label="Non-Public Schools")

# Plot open spaces (Light Green)
open_space_gdf.plot(ax=ax, color='lightgreen', alpha=0.5, label="Open Spaces")

# Add basemap (Gray-White Light Theme)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12)  

# Custom legend to ensure open spaces are represented
legend_handles = [
    mpatches.Patch(color='red', label="Syringe Requests"),
    mpatches.Patch(color='blue', label="Public Schools"),
    mpatches.Patch(color='green', label="Non-Public Schools"),
    mpatches.Patch(color='lightgreen', label="Open Spaces")  # Added for clarity
]

# Formatting
ax.legend(handles=legend_handles, loc='upper right')
plt.title("Syringe Requests, Schools, and Open Spaces in Boston")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.show()

# Keep the necessary columns
columns_to_keep = [
    # Original 311 Data
    'case_enquiry_id', 'open_dt', 'closed_dt', 'case_status', 'case_title', 
    'subject', 'reason', 'type', 'neighborhood', 'location_street_name', 
    'latitude', 'longitude', 'geometry',

    # Property Data
    'Parcel_ID', 'Property_Owner', 'Year_Built', 'Parcel_Area',

    # Schools & Open Spaces
    'Nearby_Public_School', 'Nearby_NonPublic_School', 'Nearby_Open_Space'
]

syringe_request_gdf = syringe_request_gdf[columns_to_keep]

# Ensure the DataFrame is not a slice of another DataFrame
syringe_request_gdf = syringe_request_gdf.copy()

# Convert datetime columns (allowing NaT for missing values)
syringe_request_gdf['open_dt'] = pd.to_datetime(syringe_request_gdf['open_dt'], errors='coerce')
syringe_request_gdf['closed_dt'] = pd.to_datetime(syringe_request_gdf['closed_dt'], errors='coerce')

# Total needle disposal requests
total_reports = len(syringe_request_gdf)

# Count closed cases (Met Goal)
met_goal_count = syringe_request_gdf['closed_dt'].notna().sum()
not_met_goal_count = total_reports - met_goal_count
met_goal_rate = (met_goal_count / total_reports) * 100

# Compute response time **only for rows where closed_dt exists**
syringe_request_gdf['response_time'] = (syringe_request_gdf['closed_dt'] - syringe_request_gdf['open_dt']).dt.total_seconds() / 3600

# Keep `NaN` where `closed_dt` is missing (no forced removal)
average_response_time = syringe_request_gdf['response_time'].mean(skipna=True)

# Print KPI Results
print("Total needle disposal requests:", total_reports)
print(f"Cases closed (Met Goal): {met_goal_count} ({met_goal_rate:.2f}%)")
print(f"Cases still open (Not Met): {not_met_goal_count}")
print(f"Average response time (hours): {average_response_time:.2f}")

# KPI Labels and Values
kpi_labels = ["Total Reports", "Met Goal", "Not Met"]
kpi_values = [total_reports, met_goal_count, not_met_goal_count]

# Create Bar Chart
plt.figure(figsize=(8, 6))
plt.bar(kpi_labels, kpi_values, color=['blue', 'green', 'red'])
plt.xlabel("KPI Metrics")
plt.ylabel("Number of Reports")
plt.title("KPI Analysis of Needle Reporting")

# Add data labels
for i, value in enumerate(kpi_values):
    plt.text(i, value + 50, str(value), ha='center', fontsize=12)

plt.show()

# Remove extreme outliers (e.g., response times > 99th percentile)
upper_limit = np.percentile(syringe_request_gdf['response_time'].dropna(), 99)
filtered_response_times = syringe_request_gdf[syringe_request_gdf['response_time'] <= upper_limit]['response_time'].dropna()

# Plot histogram with density normalization
plt.figure(figsize=(10, 6))
sns.histplot(filtered_response_times, bins=30, kde=False, color='blue', edgecolor='black', alpha=0.7, stat="density")

# Overlay KDE curve in red
sns.kdeplot(filtered_response_times, color='red', linewidth=2)

# Set x-axis to start from 0
plt.xlim(left=0)

plt.xlabel("Response Time (Hours)")
plt.ylabel("Density")
plt.title("Distribution of Response Time for Needle Reports (Outliers Removed)")

# Show plot
plt.show()
