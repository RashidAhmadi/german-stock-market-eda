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
# %%
#How did each company perform relative to its own starting price?
# ============================================================================
# 8. Normalize prices
# ============================================================================

normalized = adj_close / adj_close.iloc[0] * 100

print("\nNormalized prices — first rows:")
print(normalized.head())
# %%

fig, ax = plt.subplots(figsize=(12, 6))

for ticker in stocks:
    ax.plot(
        normalized.index,
        normalized[ticker],
        label=ticker
    )

ax.set_title("Normalized Performance of German Stocks")
ax.set_xlabel("Date")
ax.set_ylabel("Normalized Price (Start = 100)")
ax.legend()

fig.tight_layout()

output_file = (
    SCRIPT_DIR.parent
    / "figures"
    / "normalized_prices.png"
)

fig.savefig(output_file, dpi=300)

plt.show()
# %%

# ============================================================================
# 9. Daily returns
# A price tells us the current level of a stock.
# A return tells us how much the stock changed.
# The first observation is NaN because there is no previous trading day:
# ============================================================================

returns = adj_close.pct_change()

print("\nDaily returns:")
print(returns.head())

print("\nReturn statistics:")
print(returns.describe().round(4))
# %%

# Convert returns to percentages

returns_pct = returns * 100

print("\nDaily returns in %:")
print(returns_pct.head())
# %%

# ============================================================================
# 10. Distribution of daily returns
# How are daily returns distributed?
# ============================================================================

fig, axes = plt.subplots(2, 4, figsize=(14, 7))

for ax, ticker in zip(axes.flatten(), stocks):
    sns.histplot(
        returns_pct[ticker].dropna(),
        kde=True,
        ax=ax
    )

    ax.set_title(ticker)
    ax.set_xlabel("Daily Return (%)")

fig.suptitle(
    "Distribution of Daily Returns — German Stocks",
    fontsize=16
)

fig.tight_layout()

output_file = (
    SCRIPT_DIR.parent
    / "figures"
    / "returns_distribution.png"
)

fig.savefig(output_file, dpi=300)

plt.show()
# %%

# ============================================================================
# 11. Daily volatility
# ============================================================================

daily_volatility = returns.std()

print("\nDaily volatility:")
print(
    daily_volatility
    .sort_values(ascending=False)
    .round(4)
)

#Higher standard deviation of returns = greater day-to-day volatility.
# %%
# ============================================================================
# 12. Annualized volatility
#For financial analysis, daily volatility is often annualized.
# Assuming approximately 252 trading days per year:

# σannual=σdaily* sqrt(252)
# ============================================================================

annualized_volatility = returns.std() * np.sqrt(252)

print("\nAnnualized volatility:")
print(
    annualized_volatility
    .sort_values(ascending=False)
    .round(4)
)
# %%
# percentages

annualized_volatility_pct = annualized_volatility * 100

print("\nAnnualized volatility (%):")
print(
    annualized_volatility_pct
    .sort_values(ascending=False)
    .round(2)
)
# %%

# ============================================================================
# 13. Volatility comparison
# ============================================================================

volatility_sorted = annualized_volatility_pct[
    stocks
].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))

volatility_sorted.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Annualized Volatility of German Stocks")
ax.set_xlabel("Company")
ax.set_ylabel("Annualized Volatility (%)")

plt.xticks(rotation=45)

fig.tight_layout()

output_file = (
    SCRIPT_DIR.parent
    / "figures"
    / "volatility_comprison.png"
)

fig.savefig(output_file, dpi=300)

plt.show()
# %%

# ============================================================================
# 14. Correlation matrix
# Do German companies move together?
# ============================================================================

corr = returns[stocks].corr()

print("\nCorrelation matrix:")
print(corr.round(3))
# %%

# ============================================================================
# 15. Correlation heatmap
# ============================================================================

fig, ax = plt.subplots(figsize=(9, 7))

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True,
    ax=ax
)

ax.set_title("Correlation of Daily Returns")

fig.tight_layout()

output_file = (
    SCRIPT_DIR.parent
    / "figures"
    / "correlation_heatmap.png"
)

fig.savefig(output_file, dpi=300)

plt.show()
# %%

# ============================================================================
# 16. Correlation with DAX
# Which German company in our sample moves most closely with the German stock market?
# ============================================================================

dax_corr = returns[stocks].corrwith(
    returns["^GDAXI"]
)

print("\nCorrelation with DAX:")
print(
    dax_corr
    .sort_values(ascending=False)
    .round(3)
)
# %%

# ============================================================================
# 17. Rolling volatility

# Goal:
# Analyze how the risk/volatility of German stocks changes over time.
# ============================================================================

# We calculate:
#   30-day rolling volatility  -> short-term risk
#     -The first 29 observations will be NaN, because pandas needs 30 observations 
#     -to calculate the first standard deviation.
#   60-day rolling volatility  -> medium-term risk
#   250-day rolling volatility -> long-term risk
#
# Volatility is calculated from daily returns and annualized using sqrt(252).
# Approximately 252 trading days occur in one year.
# ============================================================================


#%% 17.1. Extract adjusted closing prices

# We use adjusted closing prices because they account for events such as
# dividends and stock splits.

prices = df["Adj Close"].copy()

print("\nAdjusted closing prices:")
print(prices.head())

print("\nShape:")
print(prices.shape)


#%% 17.2. Calculate daily returns

# Daily return:
#
#     r_t = (P_t / P_(t-1)) - 1
#
# pct_change() calculates this automatically.

returns = prices.pct_change()

print("\nDaily returns:")
print(returns.head())

print("\nReturn statistics:")
print(returns.describe())


#%% 17.3. Calculate rolling volatility

# Rolling standard deviation of daily returns.
#
# 30 days  -> short-term volatility
# 60 days  -> medium-term volatility
# 250 days -> approximately one trading year

rolling_vol_30 = returns.rolling(30).std()
rolling_vol_60 = returns.rolling(60).std()
rolling_vol_250 = returns.rolling(250).std()


#%% 17.4. Annualize volatility

# Daily volatility is converted into annualized volatility:
#
#     Annualized volatility = daily volatility × sqrt(252)
#
# 252 ≈ number of trading days per year.

rolling_vol_30_annualized = rolling_vol_30 * np.sqrt(252)
rolling_vol_60_annualized = rolling_vol_60 * np.sqrt(252)
rolling_vol_250_annualized = rolling_vol_250 * np.sqrt(252)


#%% 17.5. Display the latest volatility

print("\nLatest annualized volatility:")

latest_volatility = pd.DataFrame({
    "30D": rolling_vol_30_annualized.iloc[-1],
    "60D": rolling_vol_60_annualized.iloc[-1],
    "250D": rolling_vol_250_annualized.iloc[-1],
})

print(latest_volatility)


#%% 17.6. Plot 30-day rolling volatility for all companies

fig, ax = plt.subplots(figsize=(12, 6))

for ticker in rolling_vol_30_annualized.columns:
    ax.plot(
        rolling_vol_30_annualized.index,
        rolling_vol_30_annualized[ticker],
        label=ticker,
        alpha=0.8
    )

ax.set_title("30-Day Rolling Annualized Volatility")
ax.set_xlabel("Date")
ax.set_ylabel("Annualized Volatility")

# Display percentages instead of decimal numbers
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f"{y:.0%}")
)

ax.legend()

fig.tight_layout()

fig.savefig(
    "figures/rolling_volatility.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


#%% 17.7. Compare 30, 60 and 250-day volatility for SAP

# SAP is used as an example to demonstrate how the window size
# changes the volatility estimate.

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    rolling_vol_30_annualized["SAP.DE"],
    label="30-day"
)

ax.plot(
    rolling_vol_60_annualized["SAP.DE"],
    label="60-day"
)

ax.plot(
    rolling_vol_250_annualized["SAP.DE"],
    label="250-day"
)

ax.set_title("SAP: Rolling Annualized Volatility")
ax.set_xlabel("Date")
ax.set_ylabel("Annualized Volatility")

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f"{y:.0%}")
)

ax.legend()

fig.tight_layout()

plt.show()


#%% 17.8. Maximum volatility for each company

# Find the highest rolling volatility observed during the period.

max_volatility = rolling_vol_30_annualized.max()

# Find the date when the maximum occurred.

max_volatility_dates = rolling_vol_30_annualized.idxmax()

print("\nMaximum 30-day annualized volatility:")

for ticker in rolling_vol_30_annualized.columns:

    print(
        f"{ticker:8s} "
        f"{max_volatility[ticker]:7.2%} "
        f"on {max_volatility_dates[ticker].date()}"
    )


#%% 17.9. Create summary table

volatility_summary = pd.DataFrame({
    "Maximum 30D Volatility":
        rolling_vol_30_annualized.max(),

    "Maximum 60D Volatility":
        rolling_vol_60_annualized.max(),

    "Maximum 250D Volatility":
        rolling_vol_250_annualized.max(),

    "Latest 30D Volatility":
        rolling_vol_30_annualized.iloc[-1],

    "Latest 60D Volatility":
        rolling_vol_60_annualized.iloc[-1],

    "Latest 250D Volatility":
        rolling_vol_250_annualized.iloc[-1],
})

print("\nVolatility summary:")
print(
    volatility_summary
    .sort_values("Maximum 30D Volatility", ascending=False)
)


#%% 17.10. Save volatility summary

volatility_summary.to_csv(
    "Data/processed/volatility_summary.csv"
)

print("\nVolatility analysis completed.")
# %%

# ============================================================================
# 18 : Company Performance Comparison
#Question: Which German company performed best, which was most volatile, and which 
#had the best risk-adjusted performance during 2016–2025?
# ============================================================================
#%% Total return

# Total return over the complete available period:
#
#     Total Return = P_end / P_start - 1
#
# We use adjusted closing prices.

total_return = (
    prices.iloc[-1] / prices.iloc[0]
) - 1

print("\n--- Total Return ---")

for ticker in total_return.index:
    print(
        f"{ticker:8s} -> "
        f"{total_return[ticker]:8.2%}"
    )
    
"""
"The stocks do not share identical observation periods. Siemens Energy (ENR.DE)
 has available observations only from September 2020 onward. Therefore, 
 total-return comparisons across the complete dataset is not a perfectly fair comparison."
"""
# %%

#Annualized return is a better comparison.
# Rannual=((Pend/Pstart)**1/Year)-1
#%% Annualized return

# Determine number of years for each stock

years = (
    prices.apply(
        lambda s: (
            s.dropna().index[-1] -
            s.dropna().index[0]
        ).days / 365.25
    )
)

annualized_return = (
    prices.apply(
        lambda s: (
            s.dropna().iloc[-1] /
            s.dropna().iloc[0]
        )
    ) ** (1 / years)
) - 1

print("\n--- Annualized Return ---")

for ticker in annualized_return.index:
    print(
        f"{ticker:8s} -> "
        f"{annualized_return[ticker]:8.2%}"
    )
# %%

#%% Average annualized volatility
#Now calculate the average long-term volatility:
average_volatility = rolling_vol_250_annualized.mean()

print("\n--- Average 250-Day Annualized Volatility ---")

for ticker in average_volatility.index:
    print(
        f"{ticker:8s} -> "
        f"{average_volatility[ticker]:8.2%}"
    )
# %%

#%% Company performance summary

performance_summary = pd.DataFrame({
    "Total Return": total_return,
    "Annualized Return": annualized_return,
    "Average Volatility": average_volatility,
})

print("\n--- Company Performance Summary ---")

print(
    performance_summary
    .sort_values(
        "Annualized Return",
        ascending=False
    )
)
# %%
#. Remove the DAX from company ranking
company_summary = performance_summary.drop(
    index="^GDAXI",
    errors="ignore"
)

print("\n--- German Companies Only ---")
print(
    company_summary.sort_values(
        "Annualized Return",
        ascending=False
    )
)
#%% Annualized return comparison

fig, ax = plt.subplots(figsize=(10, 6))

sorted_returns = (
    company_summary["Annualized Return"]
    .sort_values(ascending=False)
)

sorted_returns.plot(
    kind="bar",
    ax=ax
)

ax.set_title(
    "Annualized Return of Selected German Companies"
)

ax.set_xlabel("Company")
ax.set_ylabel("Annualized Return")

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(
        lambda y, _: f"{y:.0%}"
    )
)

ax.axhline(
    0,
    linewidth=1
)

fig.tight_layout()

fig.savefig(
    "figures/annualized_returns.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# %%

