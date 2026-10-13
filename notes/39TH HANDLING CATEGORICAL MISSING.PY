# ============================================================
# CATEGORICAL IMPUTATION
# ============================================================

# ============================================================
# WHAT IS IT?
#
# When you have missing values in a CATEGORICAL column,
# you have two main approaches. Both use SimpleImputer
# but with different strategies.
# ============================================================


# ============================================================
# 1. FREQUENT CATEGORY IMPUTATION (MODE)
#
# This is the categorical version of "Mean Imputation."
# Replace missing values with the MODE — the most frequent
# category in that column.
#
# WHEN TO USE:
# → Missing data is small (less than 5%)
# → Data is missing at random (MCAR)
#
# RISK:
# If one category is already very dominant, this makes the
# dataset even MORE biased toward that category.
#
# DONT USE THIS APPROACH WHEN TWO CATEGORIES HAVE EQUAL OR
# MATCHABLE DISTRIBUTION. U CAN CHECK THAT BY PLOTTING A
# HISTOGRAM.
#
# VISUAL — when Mode imputation is DANGEROUS:
#
# BEFORE:                    AFTER mode fill:
# NY  → 50%                  NY  → 65%  ← inflated! ❌
# LA  → 30%                  LA  → 30%
# CHI → 15%                  CHI → 5%
# NaN →  5%                  (NaN filled with NY)
#
# BEFORE (balanced — DON'T use mode here):
# NY  → 48%
# LA  → 47%   ← nearly equal! Mode fill will break this
# NaN →  5%
# ============================================================


# ============================================================
# 2. ADDING A "MISSING" LABEL (MOST RECOMMENDED)
#
# WHEN THE MISSING DATA IS MORE LIKE 10 PERCENT, AT THAT
# TIME U REPLACE IT WITH THE MODE.
#
# Instead of guessing what the value was, you treat
# "Missing" as a brand new category. Replace NaN with
# a string like "Missing" or "Unknown".
#
# WHEN TO USE:
# → Safest and most common method
# → Works great when "missingness" itself might be important
#   Example: if a person didn't list their "Employment,"
#   maybe they are unemployed — that's useful information!
#
# BENEFIT:
# → You don't lose any information
# → You don't "invent" data that wasn't there
# → The model can LEARN from the fact that it was missing
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


# ============================================================
# STRATEGY 1 — MOST FREQUENT (MODE)
# ============================================================

df = pd.DataFrame({
    'City': ['NY', 'LA', 'NY', None, 'NY', 'CHI']
})

# Check distribution BEFORE filling — always do this first!
print("Distribution BEFORE imputation:")
print(df['City'].value_counts(normalize=True) * 100)
# If any two categories are close in % → don't use mode!

# Strategy = 'most_frequent'
imputer = SimpleImputer(strategy='most_frequent')
df['City'] = imputer.fit_transform(df[['City']])

print("\nDistribution AFTER mode imputation:")
print(df['City'].value_counts(normalize=True) * 100)
# Compare — if NY jumped a lot, mode imputation biased it


# ============================================================
# STRATEGY 2 — CONSTANT ("Missing" label)
# ============================================================

df2 = pd.DataFrame({
    'City': ['NY', 'LA', 'NY', None, 'NY', 'CHI', None, None]
})

# Strategy = 'constant' with fill_value='Missing'
imputer2 = SimpleImputer(strategy='constant',
                          fill_value='Missing')
df2['City'] = imputer2.fit_transform(df2[['City']])

print("\nAfter 'Missing' label imputation:")
print(df2['City'].value_counts())
# 'Missing' is now its own category
# Model can learn: "rows with Missing city behave differently"


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Two strategies for categorical missing values:
#    → most_frequent : fills with mode (most common category)
#    → constant      : fills with "Missing" string
#
# 2. WHEN to use most_frequent:
#    → Missing < 5%, data is MCAR
#    → One category is clearly dominant already
#    NEVER use when two categories have similar distribution
#    → check histogram/value_counts before deciding!
#
# 3. WHEN to use "Missing" label:
#    → Missing > 5-10%
#    → Missingness itself might carry meaning
#    → Safest default choice for categorical columns
#
# 4. ALWAYS check distribution before and after:
#    value_counts(normalize=True) * 100
#    If ratios changed significantly → imputation is biasing!
#
# 5. INTERVIEW TRAP:
#    Q: "How do you handle missing categorical data?"
#    Wrong: "Just fill with the most frequent value"
#    Right: "Depends on % missing and whether missingness
#    carries meaning. Under 5% and random → mode is okay.
#    Over 5% or missingness is informative → use a
#    'Missing' label so the model can learn from it."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → KNN Imputer
#    Instead of filling with a single statistic, KNN Imputer
#    finds the K most similar rows and uses their values
#    to fill the missing entry. Smarter, but slower.
# ============================================================