
import pandas as pd
import sqlite3
from pathlib import Path

print("Loading cleaned retail dataset...")

# Read cleaned CSV
df = pd.read_csv(
    "data/full_cleaned_retail.csv",
    dtype={"Customer ID": "string"},
    low_memory=False
)

# Create database folder
Path("database").mkdir(exist_ok=True)

# Connect to SQLite database
connection = sqlite3.connect(
    "database/retailiq.db"
)

# Import retail transactions into SQL
df.to_sql(
    "retail_transactions",
    connection,
    if_exists="replace",
    index=False,
    chunksize=10000
)

print("\nSQL database created successfully!")

# Verify number of records
cursor = connection.cursor()

cursor.execute(
    "SELECT COUNT(*) FROM retail_transactions"
)

total_records = cursor.fetchone()[0]

print("Records stored in SQL:", total_records)

# Close database connection
connection.close()

print("\nDATABASE CREATION COMPLETE!")
