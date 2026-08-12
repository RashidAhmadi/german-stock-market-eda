#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 09:02:18 2026

@author: Rashid Ahmadi
"""

# %%

# ============================================================================
# 02_data_cleaning.py
# Clean German stock market dataset
# ============================================================================

from pathlib import Path
import pandas as pd
# %%



# ============================================================================
# 1. Project paths
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_DIR
    / "Data"
    / "raw"
    / "german_stocks_raw.csv"
)

CLEAN_FILE = (
    PROJECT_DIR
    / "Data"
    / "processed"
    / "german_stocks_clean.csv"
)

# %%

# ============================================================================
# 2. Load raw dataset
# ============================================================================

print("Loading raw dataset...")

df = pd.read_csv(
    RAW_FILE,
    #tells Pandas:The first two rows contain the column names.
   header=[0, 1],
   index_col=0
)
#convert the first column (Date) to datetime
df.index = pd.to_datetime(df.index)

print("\nRaw dataset:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

# %%


# ============================================================================
# 3. Basic cleaning
# ============================================================================

# Make sure the index is a DatetimeIndex
df.index = pd.to_datetime(df.index)

# Sort observations chronologically
df = df.sort_index()

# %%


# Remove duplicated dates
duplicates = df.index.duplicated().sum()

print(f"\nDuplicated dates: {duplicates}")

if duplicates > 0:
    df = df[~df.index.duplicated(keep="first")]

# %%

# ============================================================================
# 4. Check missing values
# ============================================================================

print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False))

# %%

# 4b. Missing values by ticker
# ============================================================================

print("\nMissing values by ticker:")

tickers = df.columns.get_level_values("Ticker").unique()

for ticker in tickers:
    ticker_missing = df.xs(ticker, level="Ticker", axis=1).isna().sum().sum()

    print(f"{ticker:8} -> {ticker_missing:5} missing values")
    
# %%

# ============================================================================
# 4c. ENR.DE data availability
#Find the actual ENR.DE trading history
# ============================================================================

enr = df.xs("ENR.DE", level="Ticker", axis=1)

print("\nENR.DE date availability:")

print("First valid date:")
print(enr.dropna(how="all").index.min())

print("\nLast valid date:")
print(enr.dropna(how="all").index.max())

# %%

# ============================================================================
# 4d. Missing percentage by ticker
# ============================================================================

print("\nMissing percentage by ticker:")

for ticker in tickers:

    ticker_data = df.xs(ticker, level="Ticker", axis=1)

    missing_percentage = ticker_data.isna().mean().mean() * 100

    print(
        f"{ticker:8} -> "
        f"{missing_percentage:6.2f}% missing"
    )
# %%

# ============================================================================
# 4e. DAX missing dates
# ============================================================================

dax = df.xs("^GDAXI", level="Ticker", axis=1)

dax_missing_dates = dax[dax.isna().any(axis=1)]

print("\nDates with missing DAX observations:")
print(dax_missing_dates.index)
# %%

# ============================================================================
# 4f. Missing-value decision
# ============================================================================

print("\nMissing-value assessment:")

print(
    "ENR.DE has 47.30% missing observations and "
    "only becomes available from 2020-09-29."
)

print(
    "These missing values are treated as structural missingness "
    "and are NOT imputed."
)

print(
    "The DAX has only four missing dates, which correspond to "
    "market holidays in Germany and are therefore retained as missing."
)

# %%

# ============================================================================
# 4g. Inspect ENR.DE beginning
# ============================================================================

enr = df.xs("ENR.DE", level="Ticker", axis=1)

print("\nFirst ENR.DE observations:")
print(enr.dropna(how="all").head())

print("\nLast ENR.DE observations:")
print(enr.dropna(how="all").tail())

print("\nENR.DE observations per year:")
print(
    enr.dropna(how="all")
       .groupby(enr.dropna(how="all").index.year)
       .size()
)
# %%

# ============================================================================
# 5. Remove completely empty columns
# ============================================================================

empty_columns = df.columns[df.isna().all()]

print("\nCompletely empty columns:")
print(list(empty_columns))

if len(empty_columns) > 0: 
    df = df.drop(columns=empty_columns)


# %%

# ============================================================================
# 6. Final validation
# ============================================================================

print("\n" + "=" * 70)
print("FINAL DATA VALIDATION")
print("=" * 70)

print(f"\nShape: {df.shape}")

print("\nDate range:")
print(f"  Start: {df.index.min()}")
print(f"  End:   {df.index.max()}")

print("\nDuplicate dates:")
print(df.index.duplicated().sum())

print("\nMissing values:")
print(df.isna().sum().sum())

print("\nMissing values by ticker:")

for ticker in tickers:
    ticker_data = df.xs(ticker, level="Ticker", axis=1)
    missing = ticker_data.isna().sum().sum()

    print(f"  {ticker:8} -> {missing:5} missing values")


# %%

# ============================================================================
# 7. Save cleaned dataset
# ============================================================================

CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(CLEAN_FILE)

print("\nClean dataset saved to:")
print(CLEAN_FILE)

print("\nClean dataset shape:")
print(df.shape)

print("\nFinal date range:")
print(df.index.min(), "to", df.index.max())




# %%

