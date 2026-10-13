# ============================================================
# HANDLING MISSING DATA — COMPLETE CASE ANALYSIS (CCA)
# ============================================================

# ============================================================
# WHAT IS IT?
# TO HANDLE MISSING DATA THERE ARE THREE APPROACHES:
#
# 1. REMOVING THE ENTIRE ROW (CCA)
#    → Simply delete any row that has a missing value
#
# 2. FILLING UNIVARIATELY (Simple Imputer)
#    → Fill with Mean, Median, or Most Frequent Category
#    → Looks at ONE column at a time
#
# 3. FILLING MULTIVARIATELY (ML-based Imputers)
#    → KNN Imputer → uses K nearest neighbours to fill
#    → Iterative Imputer → uses other columns to predict
#    → Both are actual ML algorithms!
#
# THIS FILE COVERS APPROACH 1 → CCA (REMOVING DATA)
# ============================================================


# ============================================================
# WHAT IS CCA?
# ============================================================
#
# Complete Case Analysis (CCA), also known as
# "Listwise Deletion" = just THROW AWAY any row that
# contains at least ONE missing value (NaN).
#
# Simplest approach — but has REAL dangers if used wrongly!
#
# WHEN IS CCA ACTUALLY OKAY TO USE?
#
# 1. MCAR (Missing Completely At Random):
#    Data is missing PURELY BY ACCIDENT
#    Example: sensor glitch randomly lost some readings
#    If it's missing for a SPECIFIC REASON (rich people
#    refusing to report income) → CCA will BIAS your model!
#
# 2. SMALL PERCENTAGE:
#    YOU SHOULD ONLY DROP THE MISSING VALUES IF THEY ARE
#    LESS THAN 5% OF THE TOTAL VALUES
#    More than 5% → too much data lost → use imputation!
#
# 3. LARGE DATASET:
#    1 million rows, only 1000 have missing values
#    → Easier to drop than to guess (impute) those values


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load titanic or any dataset with missing values
# df = pd.read_csv('your_dataset.csv')


# ============================================================
# STEP 1 — CHECK MISSING VALUES FIRST
# ============================================================
#
# U CAN USE df.isnull().mean()*100 TO CHECK THE PERCENTAGE
# OF MISSING VALUES IN EACH COLUMN
# USING THIS U CAN CHECK THE NAMES OF THE COLUMNS WHOSE
# MISSING DATA IS LESS THAN 5 PERCENT

print("Missing value % per column:")
print(df.isnull().mean() * 100)
# .isnull()  → True/False for each cell (True = missing)
# .mean()    → average of True/False per column
#              (True=1, False=0, so mean = proportion missing)
# * 100      → convert to percentage


# ============================================================
# STEP 2 — SELECT ONLY COLUMNS WITH < 5% MISSING
# ============================================================

cols = [var for var in df.columns
        if df[var].isnull().mean() < 0.05
        and df[var].isnull().mean() > 0]
# LIST COMPREHENSION that keeps only columns where:
# missing % > 0    → has at least SOME missing values
# missing % < 0.05 → but LESS than 5% missing
#
# These are the SAFE columns to apply CCA on!

print(f"\nColumns safe to drop (< 5% missing): {cols}")


# ============================================================
# STEP 3 — APPLY CCA (DROP THE ROWS)
# ============================================================

print(f"\nShape BEFORE CCA: {df.shape}")

# Drop rows where ANY of the selected columns has NaN
new_df = df.dropna(subset=cols)

print(f"Shape AFTER  CCA: {new_df.shape}")
print(f"Rows removed    : {df.shape[0] - new_df.shape[0]}")

# dropna() PARAMETERS:
#
# df.dropna()
# → drops ANY row that has missing value in ANY column
#
# df.dropna(subset=['Age'])
# → ONLY drops row if 'Age' specifically is missing
# → keeps rows even if OTHER columns have missing values
#
# PRO TIP — TARGETED DROPPING:
# Sometimes you don't want to drop a row just because
# one UNIMPORTANT column is missing.
# Only drop if the CRITICAL column is missing!
df.dropna(subset=['Age'], inplace=True)
# inplace=True → modifies df directly without reassigning


# ============================================================
# DANGERS OF CCA — WHEN NOT TO USE IT
# ============================================================
#
# DANGER 1 → DATA LOSS CASCADE:
# If EVERY column has just a FEW missing values in
# DIFFERENT rows → you might end up deleting your
# ENTIRE DATASET! Because dropna() removes a row if
# ANY column is missing.
#
# Example:
# 100 rows, 10 columns, each has 5% missing in different rows
# → dropna() might remove 40-50% of data! ❌
#
# DANGER 2 → BIAS:
# If "Age" is missing ONLY for older people and you
# delete those rows → your model thinks everyone is young!
# CCA creates a BIASED model that fails in real world.
#
# DANGER 3 → DEPLOYMENT CRASH:
# WHEN U TRAIN YOUR MODEL U REMOVE ALL THE MISSING VALUES
# SO YOUR MODEL DOESN'T KNOW HOW TO HANDLE MISSING VALUES.
# SO THE PROGRAM CRASHES WHEN MISSING DATA IS GIVEN
# ON THE SITE WHERE THE MODEL IS DEPLOYED!
#
# SOLUTION TO DANGER 3:
# Use imputation (filling) instead of dropping
# → model is trained WITH filled values
# → at deployment, new data gets filled the same way
# → no crashes! ✅


# ============================================================
# STEP 4 — VERIFY DISTRIBUTION DIDN'T CHANGE
# ============================================================
#
# THE SHAPE OF THE DISTRIBUTION MUST NOT CHANGE AFTER
# REMOVING THE VALUES. SO YOU PLOT VALUES BEFORE AND
# AFTER REMOVING AND CHECK IF THE DISTRIBUTION IS SAME.
#
# FIRST PLOT THE ORIGINAL DATA WITH ONE COLOR AND ON THE
# SAME FIGURE PLOT THE VALUES AFTER REMOVING AND CHECK
# IF THE PREVIOUS COLOR IS STILL VISIBLE MORE OR LESS.
# THEN YOU WILL KNOW THAT THE DISTRIBUTION IS THE SAME.
#
# USE: distplot OR histogram
#
# IF THE OVERLAP IS PERFECT → values were MCAR
# (Missing Completely At Random) → safe to drop! ✅
#
# If shapes are VERY DIFFERENT → data was NOT missing
# randomly → CCA is BIASING your data → use imputation!

fig = plt.figure()
ax = fig.add_subplot(111)

# Original data (BEFORE dropping)
df['training_hours'].plot.density(color='red',
                                   label='Original')

# Data AFTER CCA
new_df['training_hours'].plot.density(color='green',
                                       label='After CCA')

ax.legend()
plt.title('Distribution Before vs After CCA\n'
          'If overlap is perfect → safe to drop!')
plt.xlabel('training_hours')
plt.show()

# HOW TO READ THIS PLOT:
# Red curve  = original distribution
# Green curve = distribution after dropping rows
# Perfect overlap → distributions are SAME → CCA is safe ✅
# Very different  → CCA is creating BIAS → DON'T use it ❌


# ============================================================
# STEP 5 — CHECKING CATEGORICAL COLUMNS AFTER CCA
# ============================================================
#
# IN CASE OF REMOVING CATEGORY COLUMNS THE RATIO SHOULD
# BE SAME FOR ALL THE QUANTITIES BEFORE AND AFTER DROPPING
#
# Example: If before CCA, 60% Male and 40% Female
# After CCA it should still be roughly 60% Male, 40% Female
# If ratio changes → CCA is biasing your categorical data!

print("\nGender distribution BEFORE CCA:")
print(df['gender'].value_counts(normalize=True) * 100)

print("\nGender distribution AFTER CCA:")
print(new_df['gender'].value_counts(normalize=True) * 100)

# normalize=True → shows proportions instead of raw counts
# * 100          → converts to percentages
# Compare both outputs → ratios should be roughly SAME ✅


# ============================================================
# CCA vs IMPUTATION — QUICK DECISION GUIDE
# ============================================================
#
# ┌─────────────────┬──────────────────────────────────────┐
# │ Situation       │ What to do                           │
# ├─────────────────┼──────────────────────────────────────┤
# │ < 5% missing    │ CCA (safe to drop) ✅               │
# │ > 5% missing    │ Use imputation (fill values)         │
# │ Missing randomly│ CCA is safe ✅                      │
# │ Missing for a   │ NEVER use CCA → biases model ❌     │
# │ specific reason │                                      │
# │ Production app  │ ALWAYS use imputation → no crashes  │
# └─────────────────┴──────────────────────────────────────┘


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. CCA = Complete Case Analysis = just drop rows with NaN
#    Simplest approach but has real dangers!
#
# 2. ONLY use CCA when:
#    → Missing < 5% of total values
#    → Data is MCAR (missing completely at random)
#    → Dataset is very large
#
# 3. Check % missing per column:
#    df.isnull().mean() * 100
#    Select safe columns: missing > 0 AND missing < 5%
#
# 4. Use subset parameter for targeted dropping:
#    df.dropna(subset=['Age']) → only drop if Age is missing
#    Don't drop whole row for one unimportant missing column!
#
# 5. ALWAYS verify distribution didn't change after CCA:
#    Plot density before (red) vs after (green) on same figure
#    Perfect overlap → MCAR → safe ✅
#    Different shapes → biased → use imputation instead ❌
#
# 6. For categorical columns → check ratio stays same
#    value_counts(normalize=True) before and after
#
# 7. NEVER use CCA for production/deployed models!
#    Model won't know how to handle missing values at runtime
#    → app crashes when user gives incomplete data ❌
#    → Always use imputation for deployed models ✅
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Univariate Imputation (Simple Imputer)
#    Instead of dropping missing values, we FILL them
#    using mean, median, or most frequent value.
#    Safe for deployment — model always gets complete data!
# ============================================================