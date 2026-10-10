# ============================================================
# CATEGORICAL ENCODING
# ============================================================

# ============================================================
# WHAT IS IT?
#
# Categorical Encoding is where we bridge the gap between
# human language and machine logic. Machines don't understand
# "Male," "Female," or "Cherbourg"; they only understand numbers.
#
# Two main types:
# → Nominal Encoding (One-Hot Encoding) — no order in data
# → Ordinal Encoding (Label Encoding)   — order exists in data
# ============================================================


# ============================================================
# 1. NOMINAL ENCODING (ONE-HOT ENCODING)
#
# Nominal data is categorical data where there is NO order or rank.
#
# Examples:
# → Gender   : Male / Female         (neither is "greater")
# → Color    : Red / Blue / Green    (no ranking)
# → Embarked : S, C, Q              (just different ports)
#
# For this, we use One-Hot Encoding (OHE).
# It creates a new binary column for each unique category.
#
# EXAMPLE — Gender column:
#
# BEFORE:               AFTER OHE:
# ┌────────┐            ┌────────────┬──────────────┐
# │ Gender │            │ Gender_Male│ Gender_Female│
# ├────────┤            ├────────────┼──────────────┤
# │  Male  │     →      │     1      │      0       │
# │ Female │            │     0      │      1       │
# │  Male  │            │     1      │      0       │
# └────────┘            └────────────┴──────────────┘
#
# THE DUMMY VARIABLE TRAP:
# If you have 3 categories, you only need 2 columns.
# Why? Because if Gender_Male=0 and Gender_Female=0,
# the machine already KNOWS it must be "Other."
# The 3rd column is REDUNDANT — it carries no new info.
# This redundancy causes multicollinearity in models.
#
# FIX → use drop='first' in sklearn to drop one column.
#
# BEFORE drop='first':       AFTER drop='first':
# ┌───┬───┬───┐              ┌───┬───┐
# │ S │ C │ Q │      →       │ C │ Q │
# ├───┼───┼───┤              ├───┼───┤
# │ 1 │ 0 │ 0 │              │ 0 │ 0 │  ← we KNOW it's S
# │ 0 │ 1 │ 0 │              │ 1 │ 0 │
# │ 0 │ 0 │ 1 │              │ 0 │ 1 │
# └───┴───┴───┘              └───┴───┘
# ============================================================


# ============================================================
# 2. ORDINAL ENCODING
#
# Ordinal data is categorical data where ORDER MATTERS.
#
# Examples:
# → Education : Bachelors < Masters < PhD
# → Review    : Poor < Good < Excellent
# → Pclass    : 1 < 2 < 3
#
# For this, we use OrdinalEncoder.
# We MANUALLY tell the machine which value is bigger.
# This is important — if you let sklearn assign numbers
# randomly, it might encode PhD=0 and Bachelors=2,
# which tells the model PhD is SMALLER than Bachelors. WRONG!
#
# EXAMPLE:
# categories=[['Poor', 'Good', 'Excellent']]
# Poor=0, Good=1, Excellent=2  ✅ correct order preserved
# ============================================================


# ============================================================
# 3. LABEL ENCODING vs ORDINAL ENCODING
#    (Very common point of confusion)
#
# ┌─────────────────┬──────────────────┬───────────────────┐
# │                 │  LabelEncoder    │  OrdinalEncoder   │
# ├─────────────────┼──────────────────┼───────────────────┤
# │ Used for        │ Target (y)       │ Features (X)      │
# │ Example         │ Yes/No → 0/1     │ Poor/Good/Exc.    │
# │ Order control   │ No (alphabetical)│ Yes (you define)  │
# │ Input shape     │ 1D array         │ 2D array          │
# └─────────────────┴──────────────────┴───────────────────┘
#
# SIMPLE RULE:
# Encoding your OUTPUT column (y)?  → LabelEncoder
# Encoding your INPUT columns (X)?  → OrdinalEncoder
# ============================================================


# ============================================================
# 4. HANDLING RARE CATEGORIES
#
# In real-world datasets, you might have a "Country" column
# where 90% of people are from India/USA, but 10 people
# are from 50 different tiny countries.
#
# If you OHE all 50 countries → 50 new columns, most filled
# with zeros. This is called a SPARSE matrix — wasteful
# and hurts model performance.
#
# FIX (CampusX Tip):
# Group all rare categories into a single "Other" label
# BEFORE encoding.
#
# threshold = 100  (or whatever makes sense for your data)
# countries with count < threshold → replace with "Other"
# Now OHE creates far fewer columns ✅
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder


# ============================================================
# SAMPLE DATA
# ============================================================

df = pd.DataFrame({
    'Gender'   : ['Male', 'Female', 'Male', 'Female'],
    'Embarked' : ['S', 'C', 'Q', 'S'],
    'Education': ['Bachelors', 'PhD', 'Masters', 'Bachelors'],
    'Country'  : ['India', 'USA', 'India', 'Bhutan'],
    'Survived' : ['Yes', 'No', 'Yes', 'No']
})


# ============================================================
# ONE-HOT ENCODING — Nominal columns
# ============================================================

ohe = OneHotEncoder(drop='first', sparse_output=False)

# Fit and transform nominal columns
encoded = ohe.fit_transform(df[['Gender', 'Embarked']])
encoded_df = pd.DataFrame(encoded,
                          columns=ohe.get_feature_names_out())

print("ONE-HOT ENCODING (drop='first'):")
print(encoded_df)
print()
# Notice: Gender has 1 column (not 2), Embarked has 2 (not 3)
# The dropped column is always recoverable — no info lost!


# ============================================================
# ORDINAL ENCODING — when order matters
# ============================================================

oe = OrdinalEncoder(
    categories=[['Bachelors', 'Masters', 'PhD']]
    # We MANUALLY define the order here — never skip this!
)

df['Education_encoded'] = oe.fit_transform(df[['Education']])

print("ORDINAL ENCODING (Education):")
print(df[['Education', 'Education_encoded']])
print()
# Bachelors=0, Masters=1, PhD=2 → correct order ✅


# ============================================================
# LABEL ENCODING — for target variable y only
# ============================================================

le = LabelEncoder()
df['Survived_encoded'] = le.fit_transform(df['Survived'])

print("LABEL ENCODING (Target Variable):")
print(df[['Survived', 'Survived_encoded']])
print()
# No/Yes → 0/1  (alphabetical order)


# ============================================================
# HANDLING RARE CATEGORIES
# ============================================================

# Group rare countries into "Other" before encoding
threshold = 2  # in real datasets this might be 100 or 1000
country_counts = df['Country'].value_counts()
df['Country_cleaned'] = df['Country'].apply(
    lambda x: x if country_counts[x] >= threshold else 'Other'
)

print("RARE CATEGORY HANDLING:")
print(df[['Country', 'Country_cleaned']])
# Bhutan appeared once → replaced with "Other"
# Now OHE won't create a useless column just for Bhutan


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. No order in data?  → One-Hot Encoding
#    Order exists?       → Ordinal Encoding
#
# 2. DUMMY VARIABLE TRAP:
#    n categories → only need n-1 columns
#    Use drop='first' in OneHotEncoder
#
# 3. LabelEncoder  → target variable y only
#    OrdinalEncoder → input features X only
#    And with OrdinalEncoder ALWAYS define category order
#    manually — never let sklearn guess the order!
#
# 4. Rare categories → group into "Other" before OHE
#    Prevents sparse matrix and useless columns
#
# 5. INTERVIEW TRAP:
#    Q: "Can you use LabelEncoder on input features?"
#    Wrong: "Yes, it converts categories to numbers"
#    Right: "Technically yes, but it assigns arbitrary numbers
#            like 0,1,2 which implies a FALSE ordering to the
#            model. Use OrdinalEncoder with defined order, or
#            OHE for nominal data."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Encoding Categorical Data
#    Converting text categories like "Male/Female" or
#    "City Names" into numbers that ML models understand.
#    Label Encoding vs One Hot Encoding — when to use which,
#    and the multicollinearity trap to avoid!
# ============================================================