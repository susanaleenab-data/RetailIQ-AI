
import pandas as pd
from pathlib import Path

print("Loading both years of retail data...")

file_path = "data/online_retail_II.xlsx"

# Load both Excel worksheets
sheets = pd.read_excel(file_path, sheet_name=None)

print("\nWorksheets found:")
for name, sheet in sheets.items():
    print(name, ":", len(sheet), "rows")

# Combine both worksheets
combined_df = pd.concat(
    sheets.values(),
    ignore_index=True
)

# Remove exact duplicates
combined_df = combined_df.drop_duplicates()

# Save the full dataset
output_path = Path("data/combined_retail.csv")
combined_df.to_csv(output_path, index=False)

print("\nTotal combined records:", len(combined_df))
print("Date range:", combined_df["InvoiceDate"].min(),
      "to", combined_df["InvoiceDate"].max())

print("\nSaved to:", output_path)
print("COMBINING DATA COMPLETE!")
