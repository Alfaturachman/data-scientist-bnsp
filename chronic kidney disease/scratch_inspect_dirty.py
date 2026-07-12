import pandas as pd
import numpy as np

df = pd.read_csv('kidney_disease.csv')

def find_non_numeric(col):
    non_num = []
    for val in df[col].dropna().unique():
        try:
            float(val)
        except ValueError:
            non_num.append(val)
    print(f"Non-numeric values in {col}: {non_num}")

find_non_numeric('pcv')
find_non_numeric('wc')
find_non_numeric('rc')

print("\n=== Unique values of DM ===")
print(df['dm'].value_counts(dropna=False))

print("\n=== Unique values of CAD ===")
print(df['cad'].value_counts(dropna=False))
