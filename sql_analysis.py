
import sqlite3
import pandas as pd

# Connect to the SQL database
conn = sqlite3.connect("database/retailiq.db")

print("\n===== RETAILIQ SQL ANALYSIS =====")

# Query 1: Total purchase records
query1 = """
SELECT COUNT(*) AS total_transactions
FROM retail_transactions;
"""

print("\n1. TOTAL PURCHASE RECORDS")
print(pd.read_sql_query(query1, conn))

# Query 2: Total purchase revenue
query2 = """
SELECT ROUND(SUM(Revenue), 2) AS total_revenue
FROM retail_transactions;
"""

print("\n2. TOTAL PURCHASE REVENUE")
print(pd.read_sql_query(query2, conn))

# Query 3: Top 10 countries by revenue
query3 = """
SELECT
    Country,
    ROUND(SUM(Revenue), 2) AS total_revenue
FROM retail_transactions
GROUP BY Country
ORDER BY total_revenue DESC
LIMIT 10;
"""

print("\n3. TOP 10 COUNTRIES BY REVENUE")
print(pd.read_sql_query(query3, conn))

# STEP 18: Advanced SQL Customer Analysis

print("\n===== TOP 10 CUSTOMERS BY REVENUE =====")

query4 = """
SELECT
    "Customer ID" AS Customer_ID,
    COUNT(DISTINCT Invoice) AS Total_Orders,
    ROUND(SUM(Revenue), 2) AS Total_Spending
FROM retail_transactions
GROUP BY "Customer ID"
ORDER BY Total_Spending DESC
LIMIT 10;
"""

print(pd.read_sql_query(query4, conn))


# Rank customers using a SQL window function
print("\n===== CUSTOMER REVENUE RANKING =====")

query5 = """
WITH CustomerSpending AS (
    SELECT
        "Customer ID" AS Customer_ID,
        ROUND(SUM(Revenue), 2) AS Total_Spending
    FROM retail_transactions
    GROUP BY "Customer ID"
)
SELECT
    Customer_ID,
    Total_Spending,
    RANK() OVER (
        ORDER BY Total_Spending DESC
    ) AS Revenue_Rank
FROM CustomerSpending
ORDER BY Revenue_Rank
LIMIT 10;
"""

print(pd.read_sql_query(query5, conn))

# STEP 19: Customer Purchase Frequency

print("\n===== CUSTOMER PURCHASE FREQUENCY =====")

query6 = """
WITH CustomerOrders AS (
    SELECT
        "Customer ID" AS Customer_ID,
        COUNT(DISTINCT Invoice) AS Total_Orders,
        ROUND(SUM(Revenue), 2) AS Total_Spending
    FROM retail_transactions
    GROUP BY "Customer ID"
)
SELECT
    CASE
        WHEN Total_Orders = 1 THEN 'One-Time'
        WHEN Total_Orders BETWEEN 2 AND 5 THEN 'Occasional'
        ELSE 'Frequent'
    END AS Customer_Segment,
    COUNT(*) AS Customer_Count,
    ROUND(SUM(Total_Spending), 2) AS Segment_Revenue
FROM CustomerOrders
GROUP BY Customer_Segment
ORDER BY Customer_Count DESC;
"""

print(pd.read_sql_query(query6, conn))

conn.close()

print("\n===== SQL ANALYSIS COMPLETE =====")
