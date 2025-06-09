-- Create the transactions table
CREATE TABLE transactions (
    "Transaction ID" VARCHAR(50) PRIMARY KEY, -- Assuming unique transaction IDs
    "Customer ID" VARCHAR(50) NOT NULL,
    "Transaction Date" DATE NOT NULL,
    "Item" VARCHAR(255) NOT NULL,
    "Category" VARCHAR(100), -- Assuming you might have this, adjust if not
    "Price Per Unit" DECIMAL(10, 2) NOT NULL,
    "Quantity" INT NOT NULL,
    "Total Spent" DECIMAL(10, 2) NOT NULL,
    "Discount Applied" BOOLEAN -- Assuming true/false for discount
);

-- Add comments for clarity 
COMMENT ON TABLE transactions IS 'Contains cleaned customer transaction data for analysis.';
COMMENT ON COLUMN transactions."Transaction ID" IS 'Unique identifier for each transaction.';
COMMENT ON COLUMN transactions."Customer ID" IS 'Unique identifier for each customer.';
COMMENT ON COLUMN transactions."Transaction Date" IS 'Date of the transaction.';
COMMENT ON COLUMN transactions."Item" IS 'Name of the product purchased.';
COMMENT ON COLUMN transactions."Category" IS 'Category of the product.';
COMMENT ON COLUMN transactions."Price Per Unit" IS 'Price of a single unit of the item.';
COMMENT ON COLUMN transactions."Quantity" IS 'Number of units purchased in the transaction.';
COMMENT ON COLUMN transactions."Total Spent" IS 'Total amount spent for the transaction (Price Per Unit * Quantity - Discount).';
COMMENT ON COLUMN transactions."Discount Applied" IS 'Boolean indicating if a discount was applied to the transaction.';

SELECT
    TO_CHAR("Transaction Date", 'YYYY-MM') AS sale_month,
    SUM("Total Spent") AS total_revenue
FROM
    transactions
GROUP BY
    sale_month
ORDER BY
    sale_month ASC
LIMIT 5;
    
SELECT
    "Item",
    SUM("Total Spent") AS total_revenue_from_item
FROM
    transactions
GROUP BY
    "Item"
ORDER BY
    total_revenue_from_item DESC
LIMIT 5;


SELECT
    "Item",
    SUM("Quantity") AS total_quantity_sold
FROM
    transactions
GROUP BY
    "Item"
ORDER BY
    total_quantity_sold DESC
LIMIT 5;

SELECT
    AVG("Total Spent") AS average_transaction_value
FROM
    transactions;
    
SELECT
    "Customer ID",
    COUNT("Transaction ID") AS number_of_transactions
FROM
    transactions
GROUP BY
    "Customer ID"
ORDER BY
    number_of_transactions DESC
LIMIT 5;

WITH ItemRevenue AS (
    SELECT
        "Item",
        SUM("Total Spent") AS item_total_revenue
    FROM
        transactions
    GROUP BY
        "Item"
),
OverallRevenue AS (
    SELECT
        SUM("Total Spent") AS overall_total_revenue
    FROM
        transactions
)
SELECT
    ir."Item",
    ir.item_total_revenue,
    (ir.item_total_revenue * 100.0 / orv.overall_total_revenue) AS percentage_of_total_revenue
FROM
    ItemRevenue ir, OverallRevenue orv
ORDER BY
    percentage_of_total_revenue DESC
LIMIT 5;

SELECT
    AVG("Quantity") AS average_quantity_per_transaction
FROM
    transactions;
    
WITH CustomerFirstPurchase AS (
    SELECT
        "Customer ID",
        MIN("Transaction Date") AS first_purchase_date
    FROM
        transactions
    GROUP BY
        "Customer ID"
)
SELECT
    TO_CHAR(first_purchase_date, 'YYYY-MM') AS acquisition_month,
    COUNT(DISTINCT "Customer ID") AS new_customers_count
FROM
    CustomerFirstPurchase
GROUP BY
    acquisition_month
ORDER BY
    acquisition_month;
    
SELECT
    "Category",
    SUM("Total Spent") AS total_revenue,
    COUNT("Transaction ID") AS number_of_transactions
FROM
    transactions
GROUP BY
    "Category"
ORDER BY
    total_revenue DESC;
    
SELECT
    "Discount Applied",
    COUNT("Transaction ID") AS number_of_transactions,
    AVG("Total Spent") AS average_spent_per_transaction,
    SUM("Total Spent") AS total_revenue
FROM
    transactions
GROUP BY
    "Discount Applied";
    
   
-- Create a view for Customer RFM metrics (Corrected for PostgreSQL date difference)
CREATE OR REPLACE VIEW customer_rfm_metrics AS
SELECT
    "Customer ID",
    MAX("Transaction Date") AS last_purchase_date,
    -- Calculate Recency in days
    -- (CURRENT_DATE - MAX("Transaction Date")) gives an INTERVAL type
    -- EXTRACT(EPOCH FROM interval) gives seconds, then divide by seconds in a day
    -- OR, more simply, just cast the interval to text and then to integer, or use AGE() and extract
    -- The most straightforward way to get total days from an interval:
    (CURRENT_DATE - MAX("Transaction Date")) AS recency_interval, -- This will show the interval (e.g., "150 days")
    EXTRACT(DAY FROM (CURRENT_DATE - MAX("Transaction Date"))) AS recency_days_from_interval, -- This gets the DAY *part* of the interval (e.g., if it's 1 month 15 days, it gets 15, not 45)
    -- *** Corrected way to get total days from an interval: ***
    -- Option 1: Use AGE() which returns an interval, then cast to integer or extract
    (EXTRACT(EPOCH FROM (CURRENT_DATE - MAX("Transaction Date"))) / (60*60*24))::INTEGER AS recency_days_total,

    -- Option 2: Directly cast the interval to integer if no months/years are involved (less robust)
    -- (CURRENT_DATE - MAX("Transaction Date"))::text::integer AS recency_days_simple,

    -- Option 3: The most robust and common way for total days:
    (CURRENT_DATE - MAX("Transaction Date")) / INTERVAL '1 day' AS recency_days_calculated,

    -- Use a fixed observation end date for consistency with Python, if applicable:
    -- ('2024-06-06'::DATE - MAX("Transaction Date")) / INTERVAL '1 day' AS recency_days_fixed_end,

    COUNT(DISTINCT "Transaction ID") AS frequency,
    SUM("Total Spent") AS monetary_value
FROM
    transactions
GROUP BY
    "Customer ID";

-- To see the data:
-- SELECT * FROM customer_rfm_metrics LIMIT 10;