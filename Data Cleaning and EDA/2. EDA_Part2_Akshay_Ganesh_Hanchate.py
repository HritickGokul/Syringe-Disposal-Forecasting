# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("2024.csv")

df

# Filter data where reason is 'Needle Program'
needle_program_df = df[df['reason'] == 'Needle Program']

needle_program_df.info()

# Remove duplicate rows
needle_program_df = needle_program_df.drop_duplicates()
needle_program_df.info()

needle_program_df.drop(columns=['sla_target_dt', 'submitted_photo'], inplace=True)

# Convert date columns to datetime format
needle_program_df['open_dt'] = pd.to_datetime(needle_program_df['open_dt'], errors='coerce')
needle_program_df['closed_dt'] = pd.to_datetime(needle_program_df['closed_dt'], errors='coerce')

# Compute resolution time in hours
needle_program_df['resolution_time_hours'] = (needle_program_df['closed_dt'] - needle_program_df['open_dt']).dt.total_seconds() / 3600

# Set plot style
sns.set(style="whitegrid")

needle_program_df['month'] = needle_program_df['open_dt'].dt.to_period('M')
monthly_trend = needle_program_df.groupby('month').size().reset_index(name='request_count')

plt.figure(figsize=(12, 6))
sns.lineplot(x=monthly_trend['month'].astype(str), y=monthly_trend['request_count'], marker="o", color="darkred")
plt.xlabel("Month")
plt.ylabel("Number of Requests")
plt.title("Monthly Trend of Syringe Requests in 2024")
plt.xticks(rotation=45)
plt.show()

# Aggregate the top 10 locations with the most cases
top_10_locations = needle_program_df['location'].value_counts().nlargest(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_locations.values, y=top_10_locations.index, palette="magma")

# Filter dataset for locations containing "INTERSECTION"
intersection_df = needle_program_df[needle_program_df['location'].str.contains("INTERSECTION", case=False, na=False)]

# Count occurrences of each intersection
intersection_counts = intersection_df['location'].value_counts().nlargest(10)

# Create bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x=intersection_counts.values, y=intersection_counts.index, palette="Reds_r")

needle_program_df.describe()

import scipy.stats as stats
# Ensure datetime format for 'open_dt'
needle_program_df['open_dt'] = pd.to_datetime(needle_program_df['open_dt'], errors='coerce')

# Create a 'weekday' column
needle_program_df['weekday'] = needle_program_df['open_dt'].dt.day_name()

# Create a new column indicating whether the request was made on a weekday or weekend
needle_program_df['is_weekend'] = needle_program_df['weekday'].isin(['Saturday', 'Sunday'])

# Extract resolution times for weekdays and weekends
weekdays_resolution = needle_program_df[needle_program_df['is_weekend'] == False]['resolution_time_hours'].dropna()
weekends_resolution = needle_program_df[needle_program_df['is_weekend'] == True]['resolution_time_hours'].dropna()

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(weekdays_resolution, weekends_resolution, equal_var=False)

print("### T-Test: Comparing Resolution Time Between Weekdays and Weekends ###")
print(f"T-Statistic: {t_stat:.2f}")
print(f"P-Value: {p_value:.4f} (Significance threshold: 0.05)")
if p_value < 0.05:
    print("Conclusion: There is a significant difference in resolution time between weekdays and weekends.")
else:
    print("Conclusion: No significant difference in resolution time between weekdays and weekends.")

# 2. Chi-Square Test: Relationship Between Case Status and Neighborhood

# Create a contingency table for case status vs. neighborhood
contingency_table = pd.crosstab(needle_program_df['case_status'], needle_program_df['neighborhood'])

# Perform chi-square test
chi2_stat, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)

# Display results
print("\n### Chi-Square Test: Relationship Between Case Status and Neighborhood ###")
print(f"Chi-Square Statistic: {chi2_stat:.2f}")
print(f"P-Value: {p_value_chi2:.4f} (Significance threshold: 0.05)")
if p_value_chi2 < 0.05:
    print("Conclusion: There is a significant relationship between case status and neighborhood.")
else:
    print("Conclusion: No significant relationship between case status and neighborhood.")

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Aggregate data to get daily counts of syringe requests
daily_requests = needle_program_df.resample('D').size()

# Fill missing values with zeros
daily_requests = daily_requests.fillna(0)

# Plot the cleaned time series data
plt.figure(figsize=(12, 6))
plt.plot(daily_requests, color='blue', marker='o', linestyle='dashed', alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Number of Requests")
plt.title("Daily Syringe Requests Over Time")
plt.grid(True)
plt.show()

# Split data into training and testing sets (80% train, 20% test)
train_size = int(len(daily_requests) * 0.8)
train, test = daily_requests[:train_size], daily_requests[train_size:]

# Fit a Seasonal ARIMA model (SARIMA)
sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
sarima_result = sarima_model.fit()

# Forecast for the test period
forecast_steps = len(test)
forecast = sarima_result.get_forecast(steps=forecast_steps)
forecast_values = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Ensure confidence intervals are numeric
forecast_ci = forecast_ci.astype(float)

# Plot the actual vs. forecasted values
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Training Data", color="blue")
plt.plot(test.index, test, label="Actual Test Data", color="green")
plt.plot(test.index, forecast_values, label="SARIMA Forecast", color="red", linestyle="dashed")
plt.fill_between(test.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='red', alpha=0.2)
plt.xlabel("Date")
plt.ylabel("Number of Requests")
plt.title("SARIMA Model - Forecast of Syringe Requests")
plt.legend()
plt.grid(True)
plt.show()

# Print SARIMA model summary
sarima_result.summary()
