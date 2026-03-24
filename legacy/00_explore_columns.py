"""
explore_columns.py — gyors oszlop áttekintő
"""
import pandas as pd

df = pd.read_csv("zerve_events.csv", low_memory=False, nrows=50000)  # csak első 50k sor, gyors

total = len(df)
print(f"Sorok (sample): {total:,}  |  Oszlopok: {df.shape[1]}\n")
print(f"{'Oszlop':<50} {'Nem-null':>10} {'Kitöltés%':>10}  {'Példa érték'}")
print("─" * 100)

for col in sorted(df.columns):
    non_null = df[col].notna().sum()
    pct      = non_null / total * 100
    sample   = df[col].dropna().iloc[0] if non_null > 0 else "—"
    sample   = str(sample)[:40]
    print(f"{col:<50} {non_null:>10,} {pct:>9.1f}%  {sample}")