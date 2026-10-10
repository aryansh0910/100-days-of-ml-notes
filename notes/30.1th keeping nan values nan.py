# ============================================================
# ENCODING WITH NaN — PRESERVING MISSING VALUES
# ============================================================

# ============================================================
# WHAT IS IT?
#
# It is very important to keep the NaN values NaN while
# encoding using any technique so that we can use that to
# fill the missing values in a proper way. But as we know,
# our model to fill the missing values would only understand
# numbers. So first we have to encode, then impute.
#
# THE ORDER MATTERS:
#
# WRONG order:
# Impute first → encode → model loses NaN info
#
# CORRECT order:
#  Raw Data (with NaNs)
#       ↓
#  Encode (keep NaNs as NaN)
#       ↓
#  Impute (MICE/IterativeImputer fills NaN smartly)
#       ↓
#  Model
#
# WHY? Because imputers like MICE work on numbers.
# If you encode first AND preserve NaN, MICE can then
# fill that NaN using patterns from other columns.
# If you fill NaN before encoding, you lose the
# information about WHAT was missing and WHY.
# ============================================================


# ============================================================
# 1. ORDINAL ENCODER (PRESERVING NaNs)
#
# This is the most common setup when you want to follow up
# with MICE (IterativeImputer). You want PhD to become 3,
# but NaN to stay NaN.
# ============================================================

from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# Define the encoder
oe = OrdinalEncoder(
    handle_unknown='use_encoded_value', # If it sees a new category, don't crash
    unknown_value=np.nan,               # Set unknown categories to NaN
    encoded_missing_value=np.nan        # CRITICAL: Keep existing NaNs as NaNs
)

# Example usage
# X[['Education']] = oe.fit_transform(X[['Education']])

# THE LOGIC:
# encoded_missing_value=np.nan is the "magic" parameter.
# By default, older versions of Scikit-Learn would crash or
# rank NaN as a number.
# This tells it: "If you see a hole, leave it as a hole."
#
# handle_unknown='use_encoded_value' + unknown_value=np.nan:
# If test data has a category the encoder never saw in
# training → don't crash, just mark it NaN.
# Then MICE handles it like any other missing value.


# ============================================================
# VISUAL — WHAT HAPPENS AT EACH STEP
#
# Raw data:
# ┌───────────┐
# │ Education │
# ├───────────┤
# │ Bachelors │
# │   NaN     │
# │  Masters  │
# │   PhD     │
# └───────────┘
#
# After OrdinalEncoder (encoded_missing_value=np.nan):
# ┌───────────┐
# │ Education │
# ├───────────┤
# │    0.0    │  ← Bachelors
# │    NaN    │  ← still NaN ✅
# │    1.0    │  ← Masters
# │    2.0    │  ← PhD
# └───────────┘
#
# After MICE Imputer:
# ┌───────────┐
# │ Education │
# ├───────────┤
# │    0.0    │
# │    1.0    │  ← NaN filled smartly using other columns
# │    1.0    │
# │    2.0    │
# └───────────┘
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Always encode FIRST, impute SECOND.
#    Never impute before encoding — you lose the missing
#    value pattern.
#
# 2. Three parameters to remember for NaN-safe encoding:
#    → encoded_missing_value=np.nan  : keep existing NaNs
#    → handle_unknown='use_encoded_value' : don't crash on
#      unseen categories in test data
#    → unknown_value=np.nan          : unseen = NaN too
#
# 3. encoded_missing_value=np.nan is the critical one.
#    Without it, sklearn may assign NaN a rank number,
#    which completely breaks your imputation step.
#
# 4. INTERVIEW TRAP:
#    Q: "Should you impute missing values before or after
#        encoding?"
#    Right answer: "Encode first, preserving NaNs.
#    Then impute. This way the imputer works on numbers
#    and can use patterns from other columns to fill NaN
#    intelligently."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → One Hot Encoding in depth
#    Handling nominal categories with no order,
#    the dummy variable trap, and drop='first' fix.
# ============================================================