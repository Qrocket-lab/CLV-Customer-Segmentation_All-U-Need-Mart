# CLV-Customer-Segmentation_All-U-Need-Mart
Customer segmentation and CLV forecasting for All-U-Need Mart, leveraging RFM analysis and predictive modeling for data-driven business recommendations.

# CLV-Customer-Segmentation-All-U-Need-Mart

## Project Overview

This project focuses on a comprehensive customer analysis for **All-U-Need Mart**, a retail company operating three branches in Myanmar. The primary objective is to gain actionable insights from past transaction data to inform strategic decision-making for the upcoming quarter.

As a Business Intelligence/Data Analyst, this initiative involved end-to-end data processing, including cleaning, exploratory data analysis (EDA), customer segmentation using RFM (Recency, Frequency, Monetary) analysis, and Customer Lifetime Value (CLV) forecasting. The insights were then visualized using Tableau and further explored for quick insights using MySQL.

## Business Problem

All-U-Need Mart's management needs a clearer understanding of its customer base to optimize marketing efforts, enhance customer retention, and drive revenue growth. The lack of granular customer insights leads to sub-optimal resource allocation and a reactive approach to customer management.

## Objectives

1.  **Identify Distinct Customer Segments:** Categorize customers based on their purchasing behavior using RFM analysis.
2.  **Forecast Customer Lifetime Value (CLV):** Predict the future spending and profitability of customers over the next 3 months.
3.  **Provide Actionable Recommendations:** Translate RFM and CLV insights into concrete strategies for personalized marketing, retention, and sales optimization.

## Methodology

This project employs a robust data analysis pipeline:

1.  **Data Collection & Initial Assessment:** Import and perform initial checks on the `retail_store_sales.csv` dataset.
2.  **Data Cleaning & Preprocessing:**
    * Handle duplicate records.
    * Address missing values using appropriate imputation strategies (median for numericals, mode for categoricals, False for boolean `Discount Applied`).
    * Convert `Transaction Date` to datetime objects and extract time-based features (Year, Month, Day, Day of Week).
    * Export cleaned data (`cleaned_retail_store_sales_forsql.csv`) for potential SQL analysis.
3.  **Exploratory Data Analysis (EDA):**
    * Analyze transaction trends over time (monthly sales and transaction counts).
    * Examine sales distribution across different store locations.
    * Determine popularity of various payment methods.
    * Identify top customers by total spent.
    * Analyze customer distribution across locations.
    * Identify top-selling product categories and individual items.
    * Calculate average price per unit per category.
4.  **RFM (Recency, Frequency, Monetary) Analysis:**
    * Calculate Recency (days since last purchase), Frequency (number of unique transactions), and Monetary (total spent) values for each customer.
    * Assign RFM scores (1-4) using custom binning for Recency and quartile-based binning for Frequency and Monetary.
    * Segment customers into 8 distinct groups (e.g., Best Customers, Loyal Customers, At Risk, Needs Attention, Lost, Other/Hibernating) based on their combined RFM scores using a rule-based approach.
5.  **Customer Lifetime Value (CLV) Prediction:**
    * Utilize the `lifetimes` Python library.
    * Fit a **Beta-Geo/Negative Binomial Distribution (BG/NBD) model** to predict the number of future purchases (frequency) for each customer.
    * Fit a **Gamma-Gamma Fitter model** (after checking for correlation between frequency and monetary value) to predict the average monetary value per transaction.
    * Calculate the **3-month Customer Lifetime Value (CLV)** by multiplying predicted purchases by predicted monetary value.
6.  **Data Export for Visualization:**
    * Combine RFM scores, segments, and CLV predictions into a single CSV file (`customer_rfm_clv_analysis.csv`) ready for advanced visualization.
7.  **Visualization & Dashboarding (Tableau):**
    * Leverage Tableau to create interactive dashboards showcasing:
        * RFM score distributions and customer segmentation.
        * Predicted CLV distribution and CLV by segment.
    * *Note: Tableau dashboards are typically shared via Tableau Public/Server or as static images/PDFs.*
8.  **Business Recommendations:**
    * Formulate actionable strategies based on the identified customer segments and their predicted lifetime value.

## Files in This Repository

* `clv_rfm_analysis.py`: The main Python script containing all data cleaning, EDA, RFM analysis, and CLV modeling code.
* `retail_store_sales.csv`: The raw transaction data for All-U-Need Mart.
* `cleaned_retail_store_sales_forsql.csv`: Cleaned dataset exported for potential SQL analysis.
* `customer_rfm_clv_analysis.csv`: Consolidated dataset with RFM scores, segments, and CLV predictions, ready for Tableau.
* `requirements.txt`: Lists all Python libraries required to run the `clv_rfm_analysis.py` script.
* `.gitignore`: Specifies intentionally untracked files to ignore.

## How to Run the Analysis

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/CLV-Customer-Segmentation-All-U-Need-Mart.git](https://github.com/YourUsername/CLV-Customer-Segmentation-All-U-Need-Mart.git)
    cd CLV-Customer-Segmentation-All-U-Need-Mart
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Place your `retail_store_sales.csv` file** in the root directory of the cloned repository.
5.  **Run the Python script:**
    ```bash
    python clv_rfm_analysis.py
    ```
    This script will perform the data cleaning, analysis, and generate the `cleaned_retail_store_sales_forsql.csv` and `customer_rfm_clv_analysis.csv` files.
6.  **For Tableau Visualization:** Open `customer_rfm_clv_analysis.csv` in Tableau Desktop and follow the steps outlined in the project documentation/slides to recreate the dashboards.
7.  **For MySQL Quick Insights:** Import `cleaned_retail_store_sales_forsql.csv` into your MySQL database and perform SQL queries as needed.

## Tools & Technologies

* **Python:** Pandas, NumPy, Matplotlib, Seaborn, `lifetimes` library.
* **Data Visualization:** Tableau
* **Database:** MySQL (for quick insights from cleaned data)
* **Version Control:** Git & GitHub

## Insights & Recommendations

(This section would typically be a more detailed summary of your findings, potentially linking to your presentation slides or static dashboard images. Since you're building a presentation, this README keeps it concise. You can elaborate here after your presentation is complete.)

* **Customer Segmentation:** Identification of 8 distinct segments (e.g., Best Customers, Loyal Customers, Needs Attention, Lost Customers) with varying behavioral patterns.
* **CLV Forecasting:** Prediction of future customer value, highlighting high-potential customers and segments.
* **Actionable Strategies:** Recommendations include targeted retention campaigns for high-value segments, re-engagement efforts for at-risk customers, and optimized marketing spend based on predicted CLV.

---
