# ============================================================
# ONE HOT ENCODING — HANDLING NaN VALUES
# ============================================================

# ============================================================
# WHAT IS IT?
#
# When you have NaN in a nominal (no-order) column and you
# want to encode it, you have two paths. Choosing the wrong
# one breaks your imputation pipeline completely.
# ============================================================


# ============================================================
# 1. THE STANDARD OHE PROBLEM
#
# By default, if OneHotEncoder sees a NaN, it creates a new
# column called x0_nan (or similar).
#
# Result:
# ┌─────────────┬───────────────┬────────────┐
# │ Gender_Male │ Gender_Female │ Gender_nan │
# ├─────────────┼───────────────┼────────────┤
# │      1      │       0       │     0      │
# │      0      │       1       │     0      │
# │      0      │       0       │     1      │  ← NaN row
# └─────────────┴───────────────┴────────────┘
#
# This tells the model "information is missing" — which is
# fine IF you don't want to impute.
#
# BUT if you want MICE to GUESS what the missing Gender
# should be, this doesn't work. Why?
# Because Gender_nan column is already filled with 1s and 0s.
# MICE sees no NaN → nothing to impute → NaN is permanently
# encoded as its own category instead of being predicted.
# ============================================================


# ============================================================
# 2. THE BETTER WAY — PLACEHOLDER STRATEGY
#
# If you want MICE to guess the missing categorical value
# (like guessing if missing Embarked should be S, C, or Q),
# use this 3-step strategy:
#
# STEP 1 → Impute a placeholder string
#           Replace NaN with 'Missing' using SimpleImputer
#
# STEP 2 → Encode using OrdinalEncoder
#           'Missing' becomes a number like any other category
#           Then use encoded_missing_value=np.nan trick
#           to convert it back to NaN
#
# STEP 3 → MICE (IterativeImputer)
#           Now sees actual NaN → predicts the correct value
#
# VISUAL:
#
# Raw:         After Step 1:    After Step 2:    After MICE:
# ┌─────────┐  ┌─────────────┐  ┌───────┐        ┌───────┐
# │Embarked │  │ Embarked    │  │Embarked│        │Embarked│
# ├─────────┤  ├─────────────┤  ├───────┤        ├───────┤
# │   S     │  │   S         │  │  2.0  │        │  2.0  │
# │   NaN   │  │  'Missing'  │  │  NaN  │  →     │  1.0  │ ← guessed
# │   C     │  │   C         │  │  0.0  │        │  0.0  │
# │   Q     │  │   Q         │  │  1.0  │        │  1.0  │
# └─────────┘  └─────────────┘  └───────┘        └───────┘
#
# WHY OrdinalEncoder and not OHE here?
# MICE works on a single column at a time.
# OHE splits one column into many → MICE gets confused
# about which column to predict.
# OrdinalEncoder keeps it as ONE column → MICE predicts
# one number → cleaner imputation.
# ============================================================


# ============================================================
# CODE
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer, IterativeImputer

df = pd.DataFrame({
    'Embarked': ['S', np.nan, 'C', 'Q', 'S', np.nan]
})

# STEP 1 → Replace NaN with placeholder string
si = SimpleImputer(strategy='constant', fill_value='Missing')
df['Embarked'] = si.fit_transform(df[['Embarked']])

# STEP 2 → Encode, treating 'Missing' as NaN after encoding
oe = OrdinalEncoder(
    categories=[['C', 'Q', 'S', 'Missing']],
    encoded_missing_value=np.nan
)
df['Embarked'] = oe.fit_transform(df[['Embarked']])
# C=0, Q=1, S=2, Missing=NaN ← back to NaN for MICE

# STEP 3 → MICE predicts the missing value
mice = IterativeImputer(random_state=42)
df['Embarked'] = mice.fit_transform(df[['Embarked']])

print(df)
# NaN is now filled with a predicted number (0, 1, or 2)
# representing the most likely Embarked value


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Default OHE creates a 'x0_nan' column for NaN.
#    This permanently encodes missingness — MICE can't
#    predict it because there's no NaN left to fill.
#
# 2. When you WANT to impute categorical NaNs:
#    SimpleImputer (placeholder) → OrdinalEncoder → MICE
#    Never OHE → MICE for this use case.
#
# 3. Why OrdinalEncoder over OHE with MICE?
#    OHE splits one column into many.
#    MICE needs ONE column with ONE NaN to predict.
#    OrdinalEncoder keeps the column intact.
#
# 4. INTERVIEW TRAP:
#    Q: "How do you impute missing values in a
#        categorical column?"
#    Wrong: "Just use SimpleImputer with most_frequent"
#    Better: "Use the placeholder strategy — SimpleImputer
#    with a string placeholder, encode with OrdinalEncoder,
#    then let MICE predict the value using patterns from
#    other columns. Much smarter than most_frequent."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → One Hot Encoding in depth
#    Handling nominal categories with no order,
#    the dummy variable trap, and drop='first' fix.
# ============================================================