# ============================================================
# FUNCTION TRANSFORMER — APPLYING CUSTOM MATH TO YOUR DATA
# ============================================================

# ============================================================
# WHAT IS IT?
# FunctionTransformer is the tool you use when built-in
# Scikit-Learn tools (like MinMaxScaler or OneHotEncoder)
# aren't enough and you need to apply your OWN CUSTOM LOGIC.
#
# IT GENERALLY MEANS APPLYING A MATHEMATICAL FORMULA
# ON YOUR DATA FOR YOUR BENEFIT.
#
# USING SUCH TRANSFORMATIONS WE CAN CONVERT OUR DATA
# INTO NORMAL DISTRIBUTION.
#
# EXAMPLES:
# → LOG TRANSFORM
# → RECIPROCAL (1/x)
# → POWER (Square Root, Cube Root)
#
# OUR MODEL LOVES NORMALLY DISTRIBUTED DATA AND PERFORMS
# BETTER DUE TO THIS. YOU CAN SEE IF YOUR DATA IS NORMALLY
# DISTRIBUTED OR NOT BY USING sns.distplot()
# ============================================================


# ============================================================
# WHEN TO USE WHICH TRANSFORM?
# ============================================================
#
# LOG TRANSFORM:
# → Use on RIGHT SKEWED data (tail on the RIGHT side)
# → Makes data close to normally distributed
# → Models like Linear Regression and Logistic Regression
#   predict BETTER on normal data
# → Very helpful in case of OUTLIERS (compresses the range)
# → Example: Salary, House Price, Fare columns
#
# X² (Square Transform):
# → Use on LEFT SKEWED data (tail on the LEFT side)
#
# SQUARE ROOT / CUBE ROOT:
# → Milder version of log transform
# → Use when log is too aggressive
#
# HOW TO CHECK WHICH SKEW YOU HAVE:
# sns.distplot(df['column'])
# Tail on RIGHT → right skewed → use LOG
# Tail on LEFT  → left skewed  → use SQUARE
# Bell curve    → already normal → no transform needed! ✅
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split


# ============================================================
# BASIC USAGE — HOW FunctionTransformer WORKS
# ============================================================
#
# FunctionTransformer(func=some_function)
# It wraps ANY Python/NumPy function so it can be used
# inside a Sklearn Pipeline like any other transformer.
#
# Think of it as: "Turn this math function into a
# Sklearn-compatible step I can plug into my Pipeline"

# ── LOG TRANSFORM ─────────────────────────────────────────
# np.log1p = log(1 + x)
# WHY log1p instead of log?
# Because log(0) = undefined (math error!)
# log1p adds 1 first so even if x=0:
# log1p(0) = log(1+0) = log(1) = 0 → safe! ✅

trf = FunctionTransformer(func=np.log1p)

# fit_transform on TRAIN only (golden rule!)
X_train_transformed = trf.fit_transform(X_train)

# transform only on TEST (no fitting!)
X_test_transformed = trf.transform(X_test)


# ── SQUARE ROOT TRANSFORM ──────────────────────────────────
trf_sqrt = FunctionTransformer(np.sqrt)
# np.sqrt is a NumPy function
# Just pass it directly — no brackets, no calling it!
# FunctionTransformer will CALL it internally on your data


# ── LOG BASE 10 TRANSFORM ─────────────────────────────────
trf_log10 = FunctionTransformer(np.log10)
# Same idea — just a different log base


# ============================================================
# THE MATHEMATICAL BENEFIT — WHY THIS ACTUALLY HELPS
# ============================================================
#
# EXAMPLE: Titanic Fare column
#
# BEFORE log transform:
# Most people paid £5-£50 (clustered at low end)
# A few people paid £500+ (extreme outliers on right)
# Range: 0 to 500+ → HUGE spread → RIGHT SKEWED
#
# AFTER log transform (log1p):
# log1p(5)   ≈ 1.79
# log1p(50)  ≈ 3.93
# log1p(500) ≈ 6.21
#
# The gap between 50 and 500 COMPRESSES from 450 to just 2.28
# Outliers are no longer "so far away" from normal values
# Distribution looks MUCH more like a bell curve → ✅
#
# RESULT: Linear Regression, Logistic Regression etc.
# find patterns MUCH easier in this compressed range!


# ============================================================
# INSIDE A PIPELINE — THE REAL USE CASE
# ============================================================
#
# The whole point of FunctionTransformer is that it works
# INSIDE a Pipeline just like any other Sklearn step.
# This is why we don't just write np.log1p(X_train) directly
# — that wouldn't work inside a Pipeline!

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('log_transform', FunctionTransformer(np.log1p)), # step 1
    ('scaler', StandardScaler()),                      # step 2
    ('model', LinearRegression())                      # step 3
])

# Now the entire pipeline:
# 1. Applies log1p to the data
# 2. Scales it with StandardScaler
# 3. Trains LinearRegression
# All in ONE pipe.fit(X_train, y_train) call! ✅


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. FunctionTransformer wraps ANY math function into
#    a Sklearn compatible transformer for use in Pipeline
#
# 2. WHEN TO USE WHICH:
#    Right skewed data → LOG TRANSFORM (np.log1p)
#    Left skewed data  → SQUARE TRANSFORM (np.square)
#    Check skewness    → sns.distplot() before deciding
#
# 3. ALWAYS use np.log1p NOT np.log
#    log1p adds 1 first → handles zero values safely
#    log(0) = undefined → your code will CRASH ❌
#
# 4. FunctionTransformer follows the SAME golden rule:
#    fit_transform(X_train) → transform(X_test)
#    NEVER fit on test data!
#
# 5. Main benefit → COMPRESSES outlier range
#    Makes data more normal → models perform better
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Power Transformer
#    A smarter, automatic version of FunctionTransformer
#    that finds the BEST mathematical transformation
#    for your data automatically using Box-Cox or Yeo-Johnson!
# ============================================================