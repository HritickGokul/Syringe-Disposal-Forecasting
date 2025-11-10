# 1. Loading and Preparing the Data

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Loading the dataset
file_path = "/Users/vidhirajeshbhaimistry/Downloads/2024.csv"
df = pd.read_csv(file_path)

# Filtering for "Needle Program" requests
df_syringe = df[df["reason"] == "Needle Program"].copy()

# Converting date columns to datetime format
df_syringe['open_dt'] = pd.to_datetime(df_syringe['open_dt'])
df_syringe['closed_dt'] = pd.to_datetime(df_syringe['closed_dt'], errors='coerce')

# Removing negative resolution times
df_syringe['resolution_time'] = (df_syringe['closed_dt'] - df_syringe['open_dt']).dt.total_seconds() / 3600
df_syringe = df_syringe[df_syringe['resolution_time'] >= 0]

# Feature Engineering: Extracting additional temporal features
df_syringe['day_of_week'] = df_syringe['open_dt'].dt.dayofweek  # Monday=0, Sunday=6
df_syringe['is_weekend'] = df_syringe['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)  # 1 if Sat/Sun, else 0

# Fix 1: Streamlined Data Processing

# Efficiently process data in one step
df_syringe = (
    df[df["reason"] == "Needle Program"]
    .assign(
        open_dt=lambda x: pd.to_datetime(x["open_dt"]),
        closed_dt=lambda x: pd.to_datetime(x["closed_dt"], errors="coerce"),
        resolution_time=lambda x: (x["closed_dt"] - x["open_dt"]).dt.total_seconds() / 3600,
        day_of_week=lambda x: x["open_dt"].dt.dayofweek,
        is_weekend=lambda x: x["open_dt"].dt.dayofweek >= 5
    )
    .query("resolution_time >= 0")  # Remove negative times efficiently
)

# Checking unique values in the 'case_status' column
unique_statuses = df_syringe['case_status'].unique()
status_counts = df_syringe['case_status'].value_counts()

# Displaying unique case statuses and their counts
print("Unique case statuses in the dataset:", unique_statuses)
print("\nCase status distribution:\n", status_counts)

# If open cases exist, extracting them; otherwise, visualizing all cases
if "OPEN" in unique_statuses:
    open_cases = df_syringe[df_syringe['case_status'] == "OPEN"]
    print("\nTotal Open Cases:", open_cases.shape[0])

    # Geospatial visualization for open cases
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=open_cases["longitude"], 
        y=open_cases["latitude"], 
        color="red", 
        alpha=0.6
    )
    plt.title("Geospatial Distribution of Open Syringe Requests in Boston")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

else:
    print("\nNo open cases found in the dataset. Using all cases for visualization instead.")

    # If no open cases exist, visualizing all cases
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df_syringe["longitude"], 
        y=df_syringe["latitude"], 
        hue=df_syringe["case_status"], 
        palette="coolwarm",
        alpha=0.5
    )
    plt.title("Geospatial Distribution of Syringe Requests in Boston")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Case Status")
    plt.show()

from sklearn.cluster import KMeans

# Preparing coordinates for clustering
coords = df_syringe[['latitude', 'longitude']].dropna()

# Applying K-Means clustering (choosing 5 clusters)
kmeans = KMeans(n_clusters=5, random_state=42)
coords["Cluster"] = kmeans.fit_predict(coords)

# Plotting clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=coords["longitude"], y=coords["latitude"], hue=coords["Cluster"], palette="coolwarm")
plt.title("K-Means Clustering of Syringe Requests")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(title="Cluster ID")
plt.show()

# Fix 2: Adding Stationarity Test Before SARIMA

from statsmodels.tsa.stattools import adfuller

# Performing ADF test
adf_test = adfuller(monthly_trends)
print("ADF Test Statistic:", adf_test[0])
print("P-Value:", adf_test[1])

# If p-value > 0.05, the series is non-stationary
if adf_test[1] > 0.05:
    print("The data is not stationary. Differencing is required before applying SARIMA.")
else:
    print("The data is stationary. Proceeding with SARIMA.")

# 2. SARIMA Time-Series Forecasting

# Aggregate requests by month for SARIMA model
df_syringe['month_year'] = df_syringe['open_dt'].dt.to_period("M")
monthly_trends = df_syringe.groupby("month_year")["case_enquiry_id"].count()

# Converting to time series format
monthly_trends.index = monthly_trends.index.to_timestamp()

# Fitting SARIMA model (p=1, d=1, q=1) with seasonality (P=1, D=1, Q=1, s=12 for monthly seasonality)
sarima_model = SARIMAX(monthly_trends, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)

# Forecasting next 6 months
forecast_periods = 6
forecast = sarima_fit.get_forecast(steps=forecast_periods)
forecast_index = pd.date_range(start=monthly_trends.index[-1], periods=forecast_periods + 1, freq="M")[1:]
forecast_values = forecast.predicted_mean

# Converting to DataFrame for visualization
forecast_df = pd.DataFrame({"Date": forecast_index, "Predicted Requests": forecast_values.values})

# Displaying Forecasted Data
print("\nSARIMA Forecasted Requests:\n", forecast_df)

# 3. Visualization: Actual vs. Forecasted Trends

# Plotting the forecast
plt.figure(figsize=(12,6))
plt.plot(monthly_trends, label="Actual Requests", marker="o")
plt.plot(forecast_index, forecast_values, label="Forecasted Requests", linestyle="dashed", color="red", marker="o")
plt.title("Forecasted Monthly Syringe Requests (SARIMA Model)")
plt.xlabel("Date")
plt.ylabel("Number of Requests")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Fix 3 : Standardizing City Council Data

# Converting to numeric, fill missing values, and group
df_syringe["city_council_district"] = pd.to_numeric(df_syringe["city_council_district"], errors="coerce").fillna(0).astype(int)

district_summary = (
    df_syringe.groupby("city_council_district")["case_enquiry_id"]
    .count()
    .reset_index()
    .rename(columns={"case_enquiry_id": "Total Requests"})
)

print("\nCleaned Syringe Requests by City Council District:\n", district_summary)

# 4. City Council District Integration & Policy Enhancements

# Grouping by City Council district if available
if "city_council_district" in df_syringe.columns:
    district_summary = df_syringe.groupby("city_council_district")["case_enquiry_id"].count().reset_index()
    district_summary.columns = ["City Council District", "Total Requests"]
    print("\nSyringe Requests by City Council District:\n", district_summary)
else:
    print("\nCity Council District data not available in dataset.")

# 5. Next Steps for Final Refinements

conclusion = """
Next Steps for Final Refinements:

1. Integrate City Council District Analysis: Enhance statistics and visualizations to align with policymaking efforts.
2. Use SARIMA Forecasting: Predict future request trends to assist Boston 311 planning.
3. Enhance Policy Implications: Clearly define how insights can help allocate resources efficiently.
4. Incorporate Heatmaps: Use geospatial analysis to visualize high-risk syringe disposal areas.
5. Finalize Power BI Dashboard: Implement refined insights, predictive analytics, and visual storytelling.

Expected Deliverables:
A. Interactive Power BI Dashboard with enhanced trends, geospatial insights, and policy recommendations.
B. Optimized Predictive Models (Linear Regression + SARIMA) for forecasting and resource planning.
C. Final Stakeholder Report with refined analysis and policy-driven insights.
D. Sponsor Presentation showcasing key findings and data-driven solutions.

"""

print(conclusion)