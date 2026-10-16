# ============================================================
# OUTLIER DETECTION & TREATMENT — Z-SCORE METHOD
# ============================================================

# ============================================================
# WHAT IS IT?
#
# If your data looks like a normal distribution (a bell
# shape), we use the Z-Score. Anything beyond 3 standard
# deviations from the mean is considered a "weirdo."
#
# THE LOGIC: If Z > 3 or Z < -3, it's an outlier.
#
# YOU CANT ALWAYS REMOVE THE OUTLIERS. YOU HAVE TO CHECK
# FIRST IF U HAVE TO REMOVE THEM. LIKE IF AGE IS SET TO BE
# 300, BUT IT CANT BE TRUE — SO YOU HAVE TO REMOVE THAT.
#
# ALGORITHMS WHERE OUTLIERS ARE A PROBLEM:
# → Linear Regression
# → Logistic Regression
# → AdaBoost
# → Deep Learning
# (these all use distance/gradient based math sensitive to
#  extreme values)
# ============================================================


# ============================================================
# WHAT CAN YOU DO ABOUT OUTLIERS?
#
# 1ST THING — TRIMMING (means removing)
#    PROBLEM → Data Lost
#
# 2ND THING — CAPPING
#    Means give the same number as the max and minimum
#    (without outliers in the dataset) to the outliers
#    Also known as Winsorization
# ============================================================


# ============================================================
# WAYS TO DETECT OUTLIERS
#
# 1. Z-SCORE FILTERING:
#    When the point in normal distribution lies beyond
#    μ + 3σ or μ - 3σ, consider them outliers
#
# 2. BOX PLOT / IQR BASED FILTERING:
#    Used in case of SKEWED distribution
#
# 3. PERCENTILE BASED:
#    Less than 2.5 percentile and more than 97.5 percentile
#
# 4. WINSORIZATION:
#    Capping values at a chosen percentile
# ============================================================


# ============================================================
# THE Z-SCORE METHOD — IN DEPTH
#
# THIS WORKS ONLY ON COLUMNS WHICH ARE NORMALLY DISTRIBUTED.
#
# The Z-score (Standard Score) tells you exactly how many
# standard deviations a specific data point is away from
# the mean.
#
# THE MATHEMATICAL LOGIC:
# When your data follows a Normal Distribution (bell curve),
# the majority of your data points (99.7%) will fall within
# 3 standard deviations. Anything outside that is a "freak
# occurrence" or an outlier.
#
# FORMULA:
# ┌─────────────────────────────────────────────┐
# │                                             │
# │           x - μ                             │
# │   Z   =  ───────                            │
# │             σ                                │
# │                                             │
# └─────────────────────────────────────────────┘
#
# THE RULE OF THUMB:
# |Z| < 1 → Very common (68% of data)
# |Z| > 2 → Unusual (only 5% of data)
# |Z| > 3 → Outlier territory
#
# In most ML projects, if Z-score > 3 or < -3, label it
# an outlier.
#
# NOW AFTER DETECTING THESE OUTLIERS, U CAN TRIM THEM OR
# U CAN CAP THEM.
#
# IF THE DATA IS SKEWED, U CANT USE THIS TECHNIQUE.
# U SHOULD PLOT A DIST PLOT TO SEE THIS — check if the
# data is actually bell-shaped before using Z-score!
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np

# Sample data for demonstration
np.random.seed(42)
df = pd.DataFrame({
    'cgpa': np.random.normal(7, 0.6, 1000),
    'placement_exam_marks': np.concatenate([
        np.random.normal(50, 15, 990),
        np.random.normal(95, 2, 10)  # outliers on high end
    ])
})


# ============================================================
# Z-SCORE METHOD — FINDING BOUNDARIES
# ============================================================

# Finding the boundary values
print("Highest allowed", df['cgpa'].mean() + 3*df['cgpa'].std())
print("Lowest allowed", df['cgpa'].mean() - 3*df['cgpa'].std())

# Highest allowed 8.808933625397177
# Lowest allowed 5.113546374602842
# UR RESULT WOULD LOOK SOMETHING LIKE THIS
# THIS WILL TELL YOU ALL THE VALUES ABOVE THIS WILL BE
# SET TO THIS OR REMOVED


# ============================================================
# FINDING THE OUTLIERS
# ============================================================

df[(df['cgpa'] > 8.80) | (df['cgpa'] < 5.11)]


# ============================================================
# TRIMMING — REMOVE THE OUTLIERS
# ============================================================

new_df = df[(df['cgpa'] < 8.80) & (df['cgpa'] > 5.11)]
new_df


# ============================================================
# CAPPING — REPLACE OUTLIERS WITH BOUNDARY VALUE
#
# NP.WHERE(CONDITION, IF TRUE, IF FALSE) — this is the
# syntax of np.where
# ============================================================

upper_limit = df['cgpa'].mean() + 3*df['cgpa'].std()
lower_limit = df['cgpa'].mean() - 3*df['cgpa'].std()

df['cgpa'] = np.where(
    df['cgpa'] > upper_limit,
    upper_limit,
    np.where(
        df['cgpa'] < lower_limit,
        lower_limit,
        df['cgpa']
    )
)
# THIS MEANS: IF THE CGPA IS MORE THAN UPPER LIMIT (TRUE),
# THEN REPLACE THAT VALUE WITH THE UPPER LIMIT.
# OTHERWISE, CHECK ANOTHER CONDITION — IF IT IS LESS THAN
# THE LOWER LIMIT, REPLACE IT WITH THE LOWER LIMIT.
# IF BOTH FALSE, DF['CGPA'] MEANS KEEP THE VALUE AS IS.

# Alternative single-condition way using .loc:
df.loc[df["cgpa"] > upper_limit, "cgpa"] = upper_limit


# ============================================================
# VISUAL — TRIMMING vs CAPPING
#
# Original: [5, 6, 7, 6, 7, 8, 7, 25]   ← 25 is an outlier
#
# After TRIMMING:
# [5, 6, 7, 6, 7, 8, 7]   ← row with 25 deleted entirely
# Row is GONE, including all other column values in that row
#
# After CAPPING (upper limit = 9):
# [5, 6, 7, 6, 7, 8, 7, 9]   ← 25 replaced with 9
# Row STAYS, only the extreme value is squashed
# ============================================================


# ============================================================
# IQR METHOD — FOR SKEWED DATA
#
# The IQR (Interquartile Range) Method is the "Bodyguard" of
# outlier detection. Unlike Z-score, which gets distracted
# by extreme values because it uses the Mean, the IQR method
# uses Percentiles. This makes it much more robust for
# skewed data — like Annual Income or CGPA — where a few
# "superstars" might exist.
#
# THE MATH BEHIND THE BOXPLOT:
# 1. Find Q1 (25th Percentile) — value below which 25% falls
# 2. Find Q3 (75th Percentile) — value below which 75% falls
# 3. Calculate IQR = Q3 - Q1 (middle 50% of your data)
#
# You'll often see people use 1.5 (for "standard" outliers)
# or 3.0 (for "extreme" outliers).
# 1.5 is the industry standard.
#
# In a normal distribution, the Q3 + 1.5×IQR boundary covers
# about 99.3% of the data. It's very similar to the "3
# Standard Deviations" rule but works much better when your
# data is leaning to one side (skewed).
#
# VISUAL — Boxplot fences:
#
#        Q1        Q3
#         |---------|
#  ───────┤█████████├──────────────── ●  ← outlier
#         |---------|
#   ←1.5×IQR    1.5×IQR→
#   Lower      Upper
#   Fence      Fence
# ============================================================

# Finding the IQR
percentile25 = df['placement_exam_marks'].quantile(0.25)
percentile75 = df['placement_exam_marks'].quantile(0.75)
iqr = percentile75 - percentile25

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print("Upper limit", upper_limit)
print("Lower limit", lower_limit)

# Upper limit 84.5
# Lower limit -23.5


# Finding Outliers
df[df['placement_exam_marks'] > upper_limit]
df[df['placement_exam_marks'] < lower_limit]
new_df = df[df['placement_exam_marks'] < upper_limit]

# SIMILARLY FOR CAPPING AS WELL
# np.where(condition, true, false)
new_df_cap = df.copy()

new_df_cap['placement_exam_marks'] = np.where(
    new_df_cap['placement_exam_marks'] > upper_limit,
    upper_limit,
    np.where(
        new_df_cap['placement_exam_marks'] < lower_limit,
        lower_limit,
        new_df_cap['placement_exam_marks']
    )
)


# ============================================================
# TRIMMING vs CAPPING — WHEN TO USE WHICH
# ============================================================

# WHEN TO APPLY TRIMMING (The "Delete" Method):
#
# 1. When your dataset is HUGE:
#    1 million rows, 100 outliers → deleting won't hurt
#
# 2. When the outliers are "Errors":
#    Age of 250, CGPA of 11/10 — clearly data entry mistakes
#    Deleting prevents the model from learning "fake" patterns
#
# 3. When you want a "Pure" Distribution:
#    Using a model extremely sensitive to outliers (Linear
#    Regression) and you have plenty of data to spare
#
# THE RISK: If your dataset is small (e.g. 500 rows),
# trimming 20 rows might delete important information in
# OTHER columns that the model needed to see.
#
# Think of it like this: Trimming is throwing away the
# "bad apples," while Capping is just slicing off the
# "bruised part" so you can still use the rest of the apple.

# WHEN TO APPLY CAPPING (The "Haircut" Method):
#
# 1. When your dataset is SMALL:
#    Can't afford to lose any rows — need every data point
#
# 2. When the outlier is "Real but Extreme":
#    For Annual_Income, a person earning $5M is a real
#    person, not an error. Don't delete them, but don't let
#    their $5M break the StandardScaler math either. Capping
#    keeps their other info (Age, City, Gender) in the game.
#
# 3. For Competition/Kaggle:
#    Usually capping performs better because it preserves
#    the sample size.
#
# ┌─────────────────┬──────────────────┬───────────────────┐
# │                 │ Trimming         │ Capping           │
# ├─────────────────┼──────────────────┼───────────────────┤
# │ What happens    │ Row deleted      │ Value squashed    │
# │ Data lost       │ Yes              │ No                │
# │ Best for        │ Errors, huge     │ Real extreme      │
# │                 │ datasets         │ values, small data│
# │ Risk            │ Losing info from │ Slight distortion │
# │                 │ other columns    │ of true values    │
# └─────────────────┴──────────────────┴───────────────────┘


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Z-score method ONLY works on normally distributed data
#    Formula: Z = (x - μ) / σ
#    |Z| > 3 → outlier
#
# 2. IQR method works on SKEWED data (uses percentiles,
#    not mean — so extreme values don't distort the boundary)
#    Upper = Q3 + 1.5×IQR, Lower = Q1 - 1.5×IQR
#
# 3. Always plot a dist plot FIRST to check if data is
#    normal (use Z-score) or skewed (use IQR)
#
# 4. TRIMMING removes the row entirely → data loss
#    CAPPING squashes the value to the boundary → no data loss
#
# 5. Trimming → huge datasets, clear data errors
#    Capping → small datasets, real extreme values, Kaggle
#
# 6. Outliers matter most for: Linear/Logistic Regression,
#    AdaBoost, Deep Learning (distance/gradient based)
#    Tree-based models (Random Forest, XGBoost) are mostly
#    unaffected by outliers
#
# 7. INTERVIEW TRAP:
#    Q: "Should you always remove outliers?"
#    Wrong: "Yes, outliers hurt model performance"
#    Right: "No — first check if it's a real value or a data
#    entry error. A CGPA of 11/10 is an error, remove it.
#    A $5M income is real, cap it instead of deleting it."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Feature Construction & Feature Splitting
#    Creating new meaningful columns by combining or
#    splitting existing ones for better model signal.
# ============================================================