#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:33:34 2026

@author: Rashid Ahmadi
"""

# %%


"""
German Stock Market EDA

Project:
Explorative Datenanalyse deutscher Unternehmen

Purpose:
- Load stock market data
- Verify data structure
- Preserve raw data
- Prepare data for subsequent EDA
"""

from pathlib import Path


import numpy as np
import yfinance as yf
import pandas as pd

# %%

# Project paths

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

RAW_DIR = PROJECT_DIR / "Data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "Data" / "processed"
FIGURES_DIR = PROJECT_DIR / "figures"

print("Project directory:")
print(PROJECT_DIR)

print("\nRaw data directory:")
print(RAW_DIR)
# %%

"""
The raw dataset contains daily historical market data for eight German companies 
and the DAX benchmark. The data returned by yfinance is structured using a pandas MultiIndex, 
with price variables as the first level and ticker symbols as the second level.
"""

tickers={
    "SAP": "SAP.DE",
   "Siemens": "SIE.DE",
   "Siemens Energy": "ENR.DE",
   "Allianz": "ALV.DE",
   "Deutsche Telekom": "DTE.DE",
   "Bayer": "BAYN.DE",
   "BASF": "BAS.DE",
   "BMW": "BMW.DE",
   "DAX": "^GDAXI"
    }

data=yf.download(
    list(tickers.values()),
    start="2016-01-01",
    end="2026-01-01",
    auto_adjust=False
    
    )

print(data.head())
# %%

print(data.shape)
print(data.columns)
# %%
#Genrating csv file 

file_path=RAW_DIR/"german_stocks_raw.csv"
data.to_csv(file_path)

# %%

# loading data from CSV

file_path = RAW_DIR / "german_stocks_raw.csv"

df = pd.read_csv(
    file_path,
    index_col=0,
    parse_dates=True
)
# %%

#checking head and tail

print(df.head())
print(df.tail())
# %%
#check shape and columns

print("shape:")
print(df.shape)
print("Columns:")
print(df.columns)

# %%
#check index, data type and print info

print(df.index)
print(df.dtypes)
print(df.info())
# %%

#Check the time range

print("Start:", data.index.min())
print("End:", data.index.max())
# %%

#check how many observations we have

print("Number of trading days:", len(data))
# %%

#Check missing values

print(data.isna().sum())
# %%

#heck the percentage too

missing_pct = data.isna().mean() * 100

print(missing_pct)
# %%

#Check duplicates
#Check the data is sorted based on index or date

print("Duplicate dates:", data.index.duplicated().sum())
print("Sorted:", data.index.is_monotonic_increasing)
# %%

