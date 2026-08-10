#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 13:33:34 2026

@author: Rashid Ahmadi
"""

# %%

import yfinance as yf
import pandas as pd

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
print(data.shape)
# %%

print(data.columns)
print(data.index)
# %%

print(data.dtypes)
print(data.info())
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

data.to_csv("german_stocks_raw.csv")