import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('kidney_disease.csv')

print("=== DataFrame Shape ===")
print(df.shape)

print("\n=== Columns and Types ===")
print(df.dtypes)

print("\n=== Null Counts ===")
print(df.isnull().sum())

print("\n=== Unique values in object/categorical columns ===")
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    print(f"{col}: {df[col].unique()[:15]} (Total unique: {df[col].nunique()})")

print("\n=== Classification column value counts ===")
print(df['classification'].value_counts(dropna=False))
