# clv_rfm_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

# Define the file path
file_path = 'retail_store_sales.csv' # If the file is in the same directory as your notebook

# Read the CSV file into a pandas DataFrame
try:
    df = pd.read_csv(file_path)
    print("Data imported successfully!")
    print("First 5 rows of the DataFrame:")
    print(df.head())
    print("\nDataFrame Info:")
    df.info()
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure the file is in the correct directory.")

# Check for duplicate rows
print("\nChecking for duplicate rows...")
print(f"Number of duplicate rows before removal: {df.duplicated().sum()}")

# Remove duplicate rows
df.drop_duplicates(inplace=True)

print(f"Number of duplicate rows after removal: {df.duplicated().sum()}")
print("Duplicates removed successfully!")
print(f"New DataFrame shape: {df.shape}")

# Check for missing values
print("\nChecking for missing values...")
print(df.isnull().sum())

# Handle missing values for numerical columns using the median
numerical_cols = ['Price Per Unit', 'Quantity', 'Total Spent']
for col in numerical_cols:
    if col in df.columns and df[col].isnull().any():
        median_value = df[col].median()
        # Best practice: Reassign the column directly instead of using inplace=True
        df[col] = df[col].fillna(median_value)
        print(f"Filled {df[col].isnull().sum()} missing values in '{col}' with median: {median_value}")

# Handle missing values for 'Item' (categorical) using the mode
if 'Item' in df.columns and df['Item'].isnull().any():
    mode_item = df['Item'].mode()[0]
    # Best practice: Reassign the column directly
    df['Item'] = df['Item'].fillna(mode_item)
    print(f"Filled {df['Item'].isnull().sum()} missing values in 'Item' with mode: '{mode_item}'")

# Handle missing values for 'Discount Applied' (boolean) with False
if 'Discount Applied' in df.columns and df['Discount Applied'].isnull().any():
    initial_discount_nan_count = df['Discount Applied'].isnull().sum()
    # Best practice: Reassign the column directly
    # Also, the Downcasting warning is handled by ensuring the dtype is appropriate
    df['Discount Applied'] = df['Discount Applied'].fillna(False).astype(bool)
    print(f"Filled {initial_discount_nan_count} missing values in 'Discount Applied' with FALSE.")

print("\nMissing values after handling:")
print(df.isnull().sum())
print("\nAll identified missing values handled successfully with best practices!")
print(f"DataFrame shape after handling missing values: {df.shape}")

# Define the output file path for the cleaned data
output_csv_file = 'cleaned_retail_store_sales_forsql.csv'

# Export the DataFrame to a CSV file
try:
    df.to_csv(output_csv_file, index=False)
    print(f"\nCleaned data successfully exported to '{output_csv_file}' for SQL analysis.")
except Exception as e:
    print(f"An error occurred while exporting the CSV: {e}")

# Display the first few rows to confirm the data is ready
print("\nFirst 5 rows of the DataFrame that will be exported:")
print(df.head())

# Set the style for plots to 'darkgrid' for dark mode compatibility
sns.set_style("darkgrid") # Changed from "whitegrid"
plt.rcParams['figure.figsize'] = (10, 6) # Set default figure size
plt.rcParams['axes.facecolor'] = '#2E2E2E' # Dark background for the plot area
plt.rcParams['figure.facecolor'] = '#2E2E2E' # Dark background for the figure
plt.rcParams['text.color'] = 'white' # White text for labels and titles
plt.rcParams['axes.labelcolor'] = 'white' # White axis labels
plt.rcParams['xtick.color'] = 'white' # White x-axis tick labels
plt.rcParams['ytick.color'] = 'white' # White y-axis tick labels
plt.rcParams['grid.color'] = '#444444' # Slightly lighter grid lines for contrast

print("--- Basic Data Overview ---")
print("\nDataFrame Info:")
df.info()

print("\nDescriptive Statistics for Numerical Columns:")
print(df.describe())

# Convert 'Transaction Date' to datetime objects
print("\nConverting 'Transaction Date' to datetime...")
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], format='%Y-%m-%d')
print("Conversion complete.")
print("\nDataFrame Info (after date conversion):")
print(df.info()) # Check info again to confirm datetime type

# Extract Year and Month for time-series analysis
df['Transaction Year'] = df['Transaction Date'].dt.year
df['Transaction Month'] = df['Transaction Date'].dt.month_name()
df['Transaction Day'] = df['Transaction Date'].dt.day_name()
df['Transaction Day_of_Week'] = df['Transaction Date'].dt.dayofweek # Monday=0, Sunday=6

print("\nFirst 5 rows with new date columns:")
print(df.head())

print("\n--- Analyzing Transaction Trends ---")
# --- 2.1 Total Transactions Over Time ---
print("\n2.1 Total Transactions Over Time (Monthly)")

# Aggregate total spent by month and year
monthly_sales = df.groupby(['Transaction Year', 'Transaction Month']).agg(
    Total_Sales=('Total Spent', 'sum'),
    Number_of_Transactions=('Transaction ID', 'nunique')
).reset_index()

monthly_sales['YearMonth'] = pd.to_datetime(monthly_sales['Transaction Year'].astype(str) + '-' +
                                             monthly_sales['Transaction Month'], format='%Y-%B')
monthly_sales = monthly_sales.sort_values('YearMonth')


plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_sales, x='YearMonth', y='Total_Sales', marker='o', color='skyblue') # Added a color that pops on dark
plt.title('Monthly Total Sales Over Time', color='white') # Ensure title is white
plt.xlabel('Date', color='white')
plt.ylabel('Total Sales', color='white')
plt.grid(True, color='#666666') # Make grid lines visible but not too bright
plt.xticks(rotation=45, color='white')
plt.yticks(color='white')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_sales, x='YearMonth', y='Number_of_Transactions', marker='o', color='lightgreen') # Added a color
plt.title('Monthly Number of Transactions Over Time', color='white')
plt.xlabel('Date', color='white')
plt.ylabel('Number of Transactions', color='white')
plt.grid(True, color='#666666')
plt.xticks(rotation=45, color='white')
plt.yticks(color='white')
plt.tight_layout()
plt.show()

# --- 2.2 Sales Distribution Across Locations ---
print("\n2.2 Sales Distribution Across Locations")
sales_by_location = df.groupby('Location')['Total Spent'].sum().sort_values(ascending=False)
print("Total Sales by Location:\n", sales_by_location)

plt.figure(figsize=(8, 6))
# FIX: Assign x variable to hue and set legend=False
sns.barplot(x=sales_by_location.index, y=sales_by_location.values, hue=sales_by_location.index, palette='flare_r', legend=False)
plt.title('Total Sales by Location', color='white')
plt.xlabel('Location', color='white')
plt.ylabel('Total Sales', color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()

# --- 2.3 Payment Method Popularity ---
print("\n2.3 Payment Method Popularity")
payment_method_counts = df['Payment Method'].value_counts()
print("Transaction Counts by Payment Method:\n", payment_method_counts)

plt.figure(figsize=(10, 7))
# Using a contrasting color palette for pie chart
plt.pie(payment_method_counts, labels=payment_method_counts.index, autopct='%1.1f%%', startangle=140,
        colors=sns.color_palette('tab20c')) # 'tab20c' or 'Set3' can be good for categorical
plt.title('Distribution of Payment Methods', color='white')
plt.axis('equal')
plt.show()

print("\n--- Analyzing Customer Trends ---")
# --- 3.1 Top Customers by Total Spent ---
print("\n3.1 Top 10 Customers by Total Spent")
top_customers = df.groupby('Customer ID')['Total Spent'].sum().nlargest(10)
print("Top 10 Customers:\n", top_customers)

plt.figure(figsize=(12, 7))
sns.barplot(x=top_customers.index, y=top_customers.values, hue=top_customers.index, palette='crest_r', legend=False)
plt.title('Top 10 Customers by Total Spent', color='white')
plt.xlabel('Customer ID', color='white')
plt.ylabel('Total Spent', color='white')
plt.xticks(rotation=45, ha='right', color='white')
plt.yticks(color='white')
plt.tight_layout()
plt.show()

# --- 3.2 Customer Distribution Across Locations ---
print("\n3.2 Customer Distribution Across Locations")
customer_count_by_location = df.groupby('Location')['Customer ID'].nunique().sort_values(ascending=False)
print("Unique Customer Count by Location:\n", customer_count_by_location)

plt.figure(figsize=(8, 6))
sns.barplot(x=customer_count_by_location.index, y=customer_count_by_location.values,
            hue=customer_count_by_location.index, palette='mako', legend=False)
plt.title('Unique Customer Count by Location', color='white')
plt.xlabel('Location', color='white')
plt.ylabel('Number of Unique Customers', color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()

print("\n--- Analyzing Product Trends ---")
# --- 4.1 Top-Selling Categories by Total Spent ---
print("\n4.1 Top 10 Selling Categories by Total Spent")
top_categories_sales = df.groupby('Category')['Total Spent'].sum().nlargest(10).sort_values(ascending=True)
print("Top 10 Categories by Sales:\n", top_categories_sales)

plt.figure(figsize=(12, 8))
# Use horizontal bar chart for better readability of category names
sns.barplot(x=top_categories_sales.values, y=top_categories_sales.index,
            hue=top_categories_sales.index, palette='rocket', legend=False)
plt.title('Top 10 Categories by Total Sales', color='white')
plt.xlabel('Total Sales', color='white')
plt.ylabel('Category', color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.tight_layout()
plt.show()

# --- 4.2 Top-Selling Individual Items by Quantity ---
print("\n4.2 Top 10 Selling Individual Items by Quantity")
top_items_quantity = df.groupby('Item')['Quantity'].sum().nlargest(10).sort_values(ascending=True)
print("Top 10 Items by Quantity Sold:\n", top_items_quantity)

plt.figure(figsize=(12, 8))
sns.barplot(x=top_items_quantity.values, y=top_items_quantity.index,
            hue=top_items_quantity.index, palette='viridis_r', legend=False)
plt.title('Top 10 Items by Quantity Sold', color='white')
plt.xlabel('Total Quantity Sold', color='white')
plt.ylabel('Item', color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.tight_layout()
plt.show()

# --- 4.3 Average Price Per Unit Per Category ---
print("\n4.3 Average Price Per Unit Per Category")
avg_price_per_category = df.groupby('Category')['Price Per Unit'].mean().sort_values(ascending=False)
print("Average Price Per Unit by Category:\n", avg_price_per_category)

plt.figure(figsize=(12, 8))
sns.barplot(x=avg_price_per_category.values, y=avg_price_per_category.index,
            hue=avg_price_per_category.index, palette='cubehelix', legend=False)
plt.title('Average Price Per Unit by Category', color='white')
plt.xlabel('Average Price Per Unit', color='white')
plt.ylabel('Category', color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.tight_layout()
plt.show()

print("\n--- RFM Analysis: Calculating R, F, M Values ---")
# Determine the snapshot date (1 day after the last transaction)
snapshot_date = df['Transaction Date'].max() + dt.timedelta(days=1)
print(f"Snapshot Date for RFM analysis: {snapshot_date.strftime('%Y-%m-%d')}")

# Group by Customer ID to calculate RFM values
rfm_df = df.groupby('Customer ID').agg(
    Recency=('Transaction Date', lambda date: (snapshot_date - date.max()).days), # Days since last purchase
    Frequency=('Transaction ID', 'nunique'),                                    # Number of unique transactions
    Monetary=('Total Spent', 'sum')                                             # Total money spent
).reset_index()

print("\nRFM DataFrame Head:")
print(rfm_df.head())

print("\nRFM DataFrame Info:")
print(rfm_df.info())

print("\nRFM DataFrame Descriptive Statistics:")
print(rfm_df.describe())

print("\n--- RFM Analysis: Assigning Scores ---")
print(rfm_df['Recency'].nunique())
recency_bins = [0, 1, 3, 5, rfm_df['Recency'].max() + 1] # Add 1 to max to ensure all values included
recency_labels = [4, 3, 2, 1] # 4=most recent, 1=least recent

rfm_df['R_Score'] = pd.cut(rfm_df['Recency'], bins=recency_bins, labels=recency_labels, right=True, include_lowest=True)
print(f"Recency Bins Used: {recency_bins}")
print(f"Recency Labels Used: {recency_labels}")

# --- Frequency Score (F_Score) ---
# Higher Frequency value gets a higher score.
# Use duplicates='drop' for robustness
rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'], q=4, labels=[1, 2, 3, 4], duplicates='drop')


# --- Monetary Score (M_Score) ---
# Higher Monetary value gets a higher score.
# Higher Monetary value gets a higher score.
# Use duplicates='drop' for robustness
rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], q=4, labels=[1, 2, 3, 4], duplicates='drop')


print("\nRFM DataFrame with Scores Head:")
print(rfm_df.head())

print("\nValue Counts for each RFM Score:")
print("R_Score Value Counts:\n", rfm_df['R_Score'].value_counts().sort_index())
print("\nF_Score Value Counts:\n", rfm_df['F_Score'].value_counts().sort_index())
print("\nM_Score Value Counts:\n", rfm_df['M_Score'].value_counts().sort_index())

# Combine RFM scores into a single RFM_Score
rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)

print("\nRFM DataFrame with Combined RFM_Score Head:")
print(rfm_df.head())

print("\n--- RFM Analysis: Segmenting Customers ---")
def rfm_segment(row):
    # Best Customers: Bought recently, buy often, and spend the most.
    if row['R_Score'] == 4 and row['F_Score'] == 4 and row['M_Score'] == 4:
        return '01 - Best Customers'
    # Loyal Customers: Buy often, good spend, but might not be most recent.
    elif row['F_Score'] == 4 and row['M_Score'] == 4:
        return '02 - Loyal Customers'
    # Big Spenders: High monetary value, regardless of recency/frequency.
    elif row['M_Score'] == 4:
        return '03 - Big Spenders'
    # Promising: Recent, but not frequent or high monetary.
    elif row['R_Score'] == 4 and row['F_Score'] >= 2: # Good recency, decent frequency
        return '04 - Promising'
    # Needs Attention: Below average recency, frequency and monetary
    elif row['R_Score'] >=2 and row['F_Score'] >=2 and row['M_Score'] >=2:
        return '05 - Needs Attention'
    # At Risk: Spent a lot and bought often, but long time ago.
    elif row['R_Score'] <= 2 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
        return '06 - At Risk'
    # Lost: Lowest scores across the board.
    elif row['R_Score'] == 1 and row['F_Score'] == 1 and row['M_Score'] == 1:
        return '07 - Lost Customers'
    # Other/Hibernating (can be refined further)
    else:
        return '08 - Other/Hibernating'


rfm_df['Segment'] = rfm_df.apply(rfm_segment, axis=1)

print("\nRFM DataFrame with Segments Head:")
print(rfm_df.head())

print("\nCustomer Segments Distribution:")
segment_counts = rfm_df['Segment'].value_counts().sort_index()
print(segment_counts)

# Visualize Segment Distribution
plt.figure(figsize=(10, 7))
sns.barplot(x=segment_counts.index, y=segment_counts.values,
            hue=segment_counts.index, palette='Spectral', legend=False)
plt.title('Distribution of Customer Segments', color='white')
plt.xlabel('Customer Segment', color='white')
plt.ylabel('Number of Customers', color='white')
plt.xticks(rotation=45, ha='right', color='white')
plt.yticks(color='white')
plt.tight_layout()
plt.show()

print("\n--- CLV Prediction: Preparing Data for Lifetimes Library ---")
# Analyze average RFM values for each segment
print("\nAverage RFM Values by Segment:")
segment_rfm_avg = rfm_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
print(segment_rfm_avg.sort_values('Recency')) # Sort by Recency to see best (lowest) first

# Import necessary models from lifetimes library
# Removed !pip install --upgrade --force-reinstall lifetimes as it's handled by requirements.txt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_history_alive
from lifetimes.utils import summary_data_from_transaction_data

print("\n--- CLV Prediction: Preparing Data for Lifetimes Library ---")
# Ensure 'Transaction Date' is datetime
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

lifetimes_df = summary_data_from_transaction_data(
    df,
    customer_id_col='Customer ID',
    datetime_col='Transaction Date',
    monetary_value_col='Total Spent',
    observation_period_end=df['Transaction Date'].max()
)

lifetimes_df.rename(columns={
    'frequency': 'frequency_ll',
    'recency': 'recency_ll',
    'T': 'T_ll'
}, inplace=True)

print("\nLifetimes Data Summary Head:")
print(lifetimes_df.head())
print("Lifetimes Data Summary Info:")
print(lifetimes_df.info())
print("Lifetimes Data Summary Descriptive Statistics:")
print(lifetimes_df.describe())

lifetimes_df_filtered = lifetimes_df[lifetimes_df['frequency_ll'] > 0]
print(f"\nNumber of customers with frequency > 0: {len(lifetimes_df_filtered)}")
print("Lifetimes Data Summary Head (Filtered for Gamma-Gamma):")
print(lifetimes_df_filtered.head())

print("\n--- CLV Prediction: Fitting BG/NBD Model ---")
bgf = BetaGeoFitter(penalizer_coef=0.1)
bgf.fit(lifetimes_df['frequency_ll'], lifetimes_df['recency_ll'], lifetimes_df['T_ll'])

print("\nBG/NBD Model Summary:")
print(bgf.summary)

# Predict future purchases for the next 3 months (approx. 90 days)
days_to_predict = 90
lifetimes_df['predicted_purchases'] = bgf.predict(
    t=days_to_predict,
    frequency=lifetimes_df['frequency_ll'],
    recency=lifetimes_df['recency_ll'],
    T=lifetimes_df['T_ll']
)

print(f"\nPredicted future purchases for the next {days_to_predict} days (approx. 3 months):")
print(lifetimes_df[['frequency_ll', 'recency_ll', 'T_ll', 'predicted_purchases']].head())

print("\nDescriptive statistics for predicted purchases:")
print(lifetimes_df['predicted_purchases'].describe())

# Step 1: Check the correlation between frequency and monetary_value
# This is an assumption of the Gamma-Gamma model.
# A low correlation (close to 0) suggests the assumption holds.
print("\nChecking correlation between frequency and monetary value for Gamma-Gamma model:")
print(lifetimes_df_filtered[['frequency_ll', 'monetary_value']].corr())

# If the correlation is high, the Gamma-Gamma model might not be appropriate.
# For low correlations, proceed with fitting.

# Initialize the GammaGammaFitter model
ggf = GammaGammaFitter(penalizer_coef=0.1) # Added a small penalizer for stability

# Fit the model to the *filtered* data (only customers with more than 1 purchase)
ggf.fit(lifetimes_df_filtered['frequency_ll'], lifetimes_df_filtered['monetary_value'])

print("\nGamma-Gamma Model Summary:")
print(ggf.summary)

# Predict the conditional expected average transaction value for each customer
# This is based on their historical spending patterns.
lifetimes_df['predicted_monetary_value'] = ggf.conditional_expected_average_profit(
    lifetimes_df['frequency_ll'],
    lifetimes_df['monetary_value']
)

print("\nPredicted average monetary value per transaction:")
print(lifetimes_df[['frequency_ll', 'monetary_value', 'predicted_monetary_value']].head())

print("\nDescriptive statistics for predicted average monetary value:")
print(lifetimes_df['predicted_monetary_value'].describe())

print("\n--- CLV Prediction: Calculating Customer Lifetime Value ---")
# Calculate CLV for the next 3 months
lifetimes_df['CLV_3_months'] = lifetimes_df['predicted_purchases'] * lifetimes_df['predicted_monetary_value']

print("\nLifetimes DataFrame with Predicted CLV Head:")
print(lifetimes_df.head())

print("\nDescriptive statistics for Predicted CLV (3 months):")
print(lifetimes_df['CLV_3_months'].describe())

# Optional: Merge CLV back to your original RFM DataFrame (rfm_df)
# This allows you to see CLV alongside RFM scores and segments.
# Make sure 'Customer ID' is the index in rfm_df or merge on the column.
# Note: The original code for this merge assumed rfm_df index was Customer ID
# The later part of the code correctly sets indexes for merging before final export
if 'rfm_df' in locals(): # Check if rfm_df exists from previous steps
    # Ensure rfm_df index is Customer ID if it's not already for this print statement
    if rfm_df.index.name != 'Customer ID':
        rfm_df.set_index('Customer ID', inplace=True)
    rfm_df = rfm_df.merge(
        lifetimes_df[['CLV_3_months']],
        left_index=True, # Assuming 'Customer ID' is the index
        right_index=True,
        how='left'
    )
    print("\nRFM DataFrame with CLV_3_months Merged (Head):")
    print(rfm_df.head())
else:
    print("\n'rfm_df' not found. CLV added to 'lifetimes_df'.")


# Visualize CLV Distribution
plt.figure(figsize=(10, 6))
sns.histplot(lifetimes_df['CLV_3_months'], bins=10, kde=True, color='skyblue', edgecolor='black')
plt.title('Distribution of Predicted Customer Lifetime Value (Next 3 Months)', color='white')
plt.xlabel('Predicted CLV (Next 3 Months)', color='white')
plt.ylabel('Number of Customers', color='white')
plt.xticks(color='white')
plt.yticks(color='white')
plt.tight_layout()
plt.show()

# Sort customers by CLV to identify your most valuable
print("\nTop 10 Customers by Predicted CLV (Next 3 Months):")
print(lifetimes_df.sort_values(by='CLV_3_months', ascending=False).head(10))

# If rfm_df is available with segments, you can analyze CLV by segment
if 'rfm_df' in locals() and 'Segment' in rfm_df.columns:
    print("\nAverage Predicted CLV by Customer Segment:")
    print(rfm_df.groupby('Segment')['CLV_3_months'].mean().sort_values(ascending=False))

# Set 'Customer ID' as index for both DataFrames if it's not already
# This part is crucial for the final merge for Tableau
if rfm_df.index.name != 'Customer ID':
    rfm_df.set_index('Customer ID', inplace=True)
if lifetimes_df.index.name != 'Customer ID':
    lifetimes_df.set_index('Customer ID', inplace=True)

print("--- Preparing Data for Tableau Export ---")
# Select relevant columns for Tableau from rfm_df
# Using R_Score, F_Score, M_Score (with capital 'S') to match your code.
rfm_cols_for_export = ['Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score', 'Segment']

# Select relevant columns for Tableau from lifetimes_df
clv_cols_for_export = ['predicted_purchases', 'predicted_monetary_value', 'CLV_3_months']

# Combine RFM and CLV data for a single customer-level dataset
# This creates the customer_analysis_df.
customer_analysis_df = rfm_df[rfm_cols_for_export].merge(
    lifetimes_df[clv_cols_for_export],
    left_index=True,
    right_index=True,
    how='left'
)

# Reset index to make 'Customer ID' a regular column for export
customer_analysis_df.reset_index(inplace=True)

print("\nCombined Customer Analysis DataFrame Head (for Tableau):")
print(customer_analysis_df.head())
print("\nCombined Customer Analysis DataFrame Info:")
print(customer_analysis_df.info())

# Export to CSV
csv_file_path_output = 'customer_rfm_clv_analysis.csv'
customer_analysis_df.to_csv(csv_file_path_output, index=False)

print(f"\nSuccessfully exported customer analysis data to: {csv_file_path_output}")