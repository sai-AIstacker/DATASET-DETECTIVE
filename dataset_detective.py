# === Dataset Detective: Automated EDA Tool ===

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os


if not os.path.exists("reports"):
    os.makedirs("reports")


sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10,6)


file_path = input("Enter CSV file path (e.g., data/sample.csv): ")
df = pd.read_csv(file_path)


print("Preview of dataset:")
print(df.head())


print("\nDataset Info:")
df.info()

print("\nBasic Statistics:")
print(df.describe(include="all"))

# 4. Missing values heatmap
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.savefig("reports/missing_values_heatmap.png")
plt.show()

# 5. Distributions of numeric columns
df.hist(bins=20, figsize=(15,10))
plt.suptitle("Numeric Column Distributions")
plt.savefig("reports/numeric_distributions.png")
plt.show()

# 6. Correlation heatmap
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("reports/correlation_heatmap.png")
plt.show()

# 7. Auto-Narrative
print("\n--- Auto-Narrative ---")
print(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
missing = df.isnull().sum().sum()
print(f"There are {missing} missing values in total.")
print(f"The column with the most missing values is '{df.isnull().sum().idxmax()}'.")
print("The highest correlation is between:")
print(corr.unstack().sort_values(ascending=False).drop_duplicates().head(2))

# 8. Auto-generate text report
report_file = "reports/report.txt"

with open(report_file, "w") as f:
    f.write("=== Dataset Detective Report ===\n\n")

    # Dataset preview
    f.write("Preview of dataset:\n")
    f.write(df.head().to_string() + "\n\n")

    # Dataset info captured properly
    buffer = io.StringIO()
    df.info(buf=buffer)
    f.write(buffer.getvalue() + "\n\n")

    # Basic statistics
    f.write("Basic Statistics:\n")
    f.write(df.describe(include="all").to_string() + "\n\n")

    # Auto-narrative
    f.write("--- Auto-Narrative ---\n")
    f.write(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n")
    f.write(f"There are {missing} missing values in total.\n")
    f.write(f"The column with the most missing values is '{df.isnull().sum().idxmax()}'.\n")
    f.write("The highest correlation is between:\n")
    f.write(str(corr.unstack().sort_values(ascending=False).drop_duplicates().head(2)) + "\n")

print(f"\n✅ Report saved successfully to: {report_file}")
