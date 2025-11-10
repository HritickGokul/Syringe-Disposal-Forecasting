Project: Boston 311 Syringe Requests Analysis Dashboard


File Naming Convention: [Filename]_[Author Name]


Folder Structure:

1. Data Cleaning and EDA
   - HTML Files (DEMO): Contains Python code output in HTML format. Open with a browser to view code and results directly.
   - Datasets (manually downloaded):
     • 2021-2025 Boston 311 Service Requests Dataset
     • Boston_Parcels_Full.csv
     • zipcodes
     • city_council_districts___2023_2032
     • public_schools.csv
     • non_public_schools.csv
     • open_space.csv
   - Python Scripts:
     • Data cleaning steps: standardized column names, removed irrelevant columns, filled missing values, filtered with "Needle Program" keyword, fixed city council district with shapefiles, spatially matched to add `owner_name`, checked for syringe requests within 50 meters of sensitive areas
     • Exploratory Data Analysis: SARIMA modeling, K-Means Clustering, Summary Tables, Bar Charts, Line Charts

2. Dashboard
   - Aggregated and formatted datasets for Power BI:
     • powerbi_boston_311_needle_program_dataset_cleaned.csv
     • powerbi_final_combined_dataset.csv
   - Power BI Dashboard File:
     • 311_Boston_Syringe_Request_Dashboard_Shengyao_Xu.pbix

3. Data API Downloading
   - HTML Files (DEMO): Python code and output display
   - Python Scripts:
     • API_Auto_Data_Download_Shengyao_Xu.py — Automates downloading 311 data via API using two different methods

4. Project Documents
   - Mid-term Presentation.pptx
   - Final Presentation.pptx
   - Final Project Report.pdf
   - Individual Contribution Report.pdf


Executive Summary

This project focuses on analyzing Boston 311 service requests related to the “Needle Program” from 2021 to 2025 in order to address the ongoing opioid crisis through data-driven insights. The project integrates multiple publicly available datasets and applies techniques such as data cleaning, spatial analysis, clustering, and time series forecasting (SARIMA) to uncover geographic and temporal patterns of syringe disposal requests.

All Python scripts generate HTML output files located in the HTML Files (DEMO) folder. These files can be opened directly in a web browser to view the full code and its execution results — no need to re-run the .py files. This allows for quick review of the analysis process and outcomes.

A Power BI dashboard was also developed using the cleaned and aggregated data to allow interactive visual analysis by city officials and stakeholders. Additionally, Python scripts were included to automate the process of downloading updated Boston 311 data via API.

⚠️Note: Users must modify the dataset file paths in the Python scripts to match their own local file system in order to re-run the scripts successfully.


Thank you!

Group 8
