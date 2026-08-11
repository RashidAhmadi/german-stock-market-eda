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
# 6. Save cleaned dataset
# ============================================================================

CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(CLEAN_FILE)

print("\nClean dataset saved to:")
print(CLEAN_FILE)

print("\nClean dataset shape:")
print(df.shape)