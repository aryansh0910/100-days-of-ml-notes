# ============================================================
# RANDOM IMPUTATION + MISSING INDICATOR
# ============================================================

# ============================================================
# WHAT IS RANDOM IMPUTATION?
#
# U MOSTLY USE THIS TECHNIQUE IN CASE OF LINEAR MODELS AS
# FOR THEM THE DISTRIBUTION MATTERS. BUT IT DOESNT WORK WELL
# IN CASE OF DECISION TREES BECAUSE IT WOULD GIVE RANDOMNESS
# TO THE DATA.
#
# ANOTHER DISADVANTAGE: U NEED TO KEEP THE ORIGINAL DATA
# WITH U TO FILL THE VALUES EVEN ON THE SERVER.
#
# Random Sample Imputation is a technique where we replace
# missing values with a randomly selected observation from
# the available (non-missing) data in that same column.
#
# HOW IT WORKS:
# 1. Take all the non-missing values in a column
# 2. Pick one at random
# 3. Fill the NaN with that picked value
# 4. Repeat for every missing cell
#
# WHY USE THIS? (The "Magic" of Distribution)
# The biggest advantage over Mean or Median is that it
# PRESERVES the variance and distribution of the data.
#
# VISUAL — why distribution preservation matters:
#
# Original Age column:
# [25, 30, 45, 50, 22, 38, 31] → spread out, realistic
#
# After MEAN fill (3 NaNs filled with mean=34):
# [25, 30, 34, 45, 50, 34, 22, 38, 34, 31]
#      ↑spike at 34 → distribution changes shape ❌
#
# After RANDOM fill (3 NaNs filled randomly from existing):
# [25, 30, 45, 45, 50, 30, 22, 38, 25, 31]
#      ↑values already existed → distribution stays same ✅
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# SAMPLE DATA
# ============================================================

data = {
    'Employee_ID': [101, 102, 103, 104, 105,
                    106, 107, 108, 109, 110],
    'Age'        : [25, 30, np.nan, 45, 50,
                    np.nan, 22, 38, np.nan, 31],
    'Department' : ['HR', 'IT', 'IT', 'Sales', np.nan,
                    'HR', 'IT', np.nan, 'Sales', 'IT'],
    'Salary'     : [50000, 60000, 62000, 58000, 70000,
                    52000, 48000, 65000, 59000, 61000]
}

df = pd.DataFrame(data)

# FIRST U ALWAYS DO TRAIN TEST SPLIT TO PREVENT DATA LEAKAGE.
# OTHERWISE UR MODEL WILL GET TO KNOW ABOUT THE VALUES IN
# TEST AS WELL.
X_train, X_test = train_test_split(df, test_size=0.2,
                                    random_state=42)
print(X_train)


# ============================================================
# RANDOM IMPUTATION — STEP BY STEP
# ============================================================

# STEP 1 → find the index positions where Age is NaN
z = X_train["Age"][X_train["Age"].isnull()]
# first "Age" → get only the Age column values
# next bracket → boolean condition to get only missing rows
print("PRINTING THE VALUES")
print(z)

# STEP 2 → fill those positions with random samples
# dropna()  → removes NaN so we don't accidentally sample NaN
# sample(n) → normally u pass a fixed number like sample(2)
#             but here we don't know how many NaNs exist
#             so we use .isnull().sum() to get exact count
# .values   → strips the index, so when we assign back
#             there's no index mismatch error
# .loc      → used instead of direct assignment to avoid
#             pandas SettingWithCopyWarning

X_train.loc[X_train["Age"].isnull(), "Age"] = (
    X_train["Age"]
    .dropna()
    .sample(X_train["Age"].isnull().sum(), random_state=42)
    .values
)

print(X_train)  # now all values are filled


# ============================================================
# MISSING INDICATOR
#
# A Missing Indicator is a technique where you create a
# binary (True/False) column to tell your ML model exactly
# WHERE the data was originally missing.
#
# BY ADDING THIS, OUR MODEL WILL GET TO KNOW WHEN TO CREATE
# A DIFFERENCE BETWEEN THE MISSING VALUES OR NOT.
#
# THE PROBLEM — "The Secret Signal":
# Sometimes the fact that data is missing is NOT an accident.
# Example: In a medical dataset, if "Blood Pressure" is
# missing, maybe the patient was too unstable to have it
# taken. That's critical information!
#
# If you just fill with Mean or Random Sample, the model
# FORGETS the data was ever missing. You've erased a
# potentially important signal.
#
# HOW IT WORKS:
# Create a new column named ColName_is_missing
# → Original value was NaN → new column gets 1 (True)
# → Original value was present → new column gets 0 (False)
# Keep BOTH: the imputed column AND the indicator column.
#
# VISUAL:
#
# BEFORE:          AFTER (add_indicator=True):
# ┌─────┐          ┌─────┬──────────────┐
# │ Age │          │ Age │ Age_is_missing│
# ├─────┤          ├─────┼──────────────┤
# │ 20  │          │ 20  │      0       │
# │ NaN │    →     │ 30  │      1       │ ← was missing
# │ 30  │          │ 30  │      0       │
# │ NaN │          │ 30  │      1       │ ← was missing
# │ 40  │          │ 40  │      0       │
# └─────┘          └─────┴──────────────┘
# Model can now learn: "rows where Age was missing
# behave differently from rows where Age was present"
# ============================================================

from sklearn.impute import SimpleImputer

X_train_ind = pd.DataFrame({'Age': [20, np.nan, 30,
                                     np.nan, 40]})

# add_indicator=True → creates the extra binary column
si = SimpleImputer(strategy='median', add_indicator=True)

X_train_transformed = si.fit_transform(X_train_ind)
X_train_transformed = pd.DataFrame(
    X_train_transformed,
    columns=["AGE", "FLAG"]
)

print(X_train_transformed)
# AGE  → filled with median
# FLAG → 1 where original was NaN, 0 where it was present


# ============================================================
# SELECTING THE BEST IMPUTATION STRATEGY — GRIDSEARCHCV
#
# Instead of guessing which strategy works best, use
# GridSearchCV to TEST all strategies and let the model
# tell you which one gives the best score.
# ============================================================

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# 1. Setup Data
X = df.drop('Salary', axis=1).select_dtypes(include='number')
y = df['Salary']
X_train_gs, X_test_gs, y_train_gs, y_test_gs = train_test_split(
    X, y, test_size=0.2
)

# 2. Create the Pipeline
# imputer first, then model — clean single workflow
pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('model', RandomForestRegressor())
])

# 3. Define the Search Space
# __ separates step name from its parameter
param_grid = {
    'imputer__strategy'     : ['mean', 'median', 'constant'],
    'imputer__add_indicator': [True, False]
}

# 4. Run the Search
# cv=5 → 5-fold cross validation
# neg_mean_squared_error → lower error = better
grid = GridSearchCV(pipe, param_grid, cv=5,
                    scoring='neg_mean_squared_error')
grid.fit(X_train_gs, y_train_gs)

# 5. See the Winner
print(f"Best Strategy: {grid.best_params_}")
# Now you KNOW which strategy + indicator combo works best
# for YOUR specific data — no guessing needed!


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Random Imputation = fill NaN with randomly picked
#    values from existing non-missing data in same column.
#    KEY BENEFIT: preserves distribution shape and variance.
#
# 2. USE for linear models where distribution matters.
#    AVOID for tree models — adds unnecessary randomness.
#
# 3. DISADVANTAGE: must store original training data on
#    server to sample from it at deployment time.
#
# 4. ALWAYS dropna() before .sample() — otherwise you
#    might randomly pick another NaN to fill a NaN!
#
# 5. Use .values after sample() to strip index —
#    prevents index mismatch when assigning back.
#    Use .loc for assignment — avoids pandas warning.
#
# 6. Missing Indicator (add_indicator=True):
#    Creates a binary FLAG column alongside imputed column.
#    Use when missingness itself is a signal — model learns
#    "rows where this was missing behave differently."
#
# 7. Use GridSearchCV + Pipeline to find the best strategy:
#    Don't guess — let cross validation tell you!
#    param name format: 'stepname__parameter'
# ============================================================


# ============================================================
# COMING UP NEXT:
# → KNN Imputer
#    Finds K most similar rows and uses their values to fill
#    the missing entry. Smarter than random, but slower.
# ============================================================