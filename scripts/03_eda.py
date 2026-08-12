#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 10:57:48 2026

@author: Rashid Ahmadi
"""

# %%

# ============================================================================
# 03_eda.py
# Exploratory Data Analysis of German Stocks
# ============================================================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"

# %%

# ============================================================================
# 1. Paths
# ============================================================================

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

CLEAN_FILE = (
    SCRIPT_DIR.parent
    / "Data"
    / "processed"
    / "german_stocks_clean.csv"
)

# %%


# ============================================================================
# 2. Load cleaned dataset
# ============================================================================

df = pd.read_csv(
    CLEAN_FILE,
    header=[0, 1],
    index_col=0,
    parse_dates=True
)

print("=" * 70)
print("GERMAN STOCK MARKET — EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nDate range:")
print(df.index.min(), "to", df.index.max())

print("\nColumn structure:")
print(df.columns)
# %%

# ============================================================================
# 3. Identify assets
# ============================================================================

tickers = df.columns.get_level_values("Ticker").unique()

print("\nTickers:")
for ticker in tickers:
    print(" ", ticker)

stocks = [ticker for ticker in tickers if ticker != "^GDAXI"]

print("\nGerman companies:")
for ticker in stocks:
    print(" ", ticker)

print("\nBenchmark:")
print(" ", "^GDAXI")
# %%

# ============================================================================
# 4. Extract adjusted closing prices
# ============================================================================

adj_close = df["Adj Close"]

print("\nAdjusted closing prices:")
print(adj_close.head())

print("\nShape:")
print(adj_close.shape)
# %%

# ============================================================================
# 5. Basic descriptive statistics
# ============================================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(adj_close.describe().round(2))
# %%

# ============================================================================
# 6. Missing values
# ============================================================================

print("\nMissing adjusted closing prices:")

missing = adj_close.isna().sum()

print(missing)
# %%

# ============================================================================
# 7. Stock price development
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

for ticker in stocks:
    ax.plot(
        adj_close.index,
        adj_close[ticker],
        label=ticker
    )

ax.set_title("German Stock Price Development (2016–2025)")
ax.set_xlabel("Date")
ax.set_ylabel("Adjusted Close Price (€)")
ax.legend()

fig.tight_layout()

output_file = (
    SCRIPT_DIR.parent
    / "figures"
    / "price_development.png"
)

fig.savefig(output_file, dpi=300)

plt.show()