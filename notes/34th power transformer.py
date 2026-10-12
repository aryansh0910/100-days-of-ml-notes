# ============================================================
# POWER TRANSFORMER — AUTOMATIC NORMALIZATION
# ============================================================

# ============================================================
# WHAT IS IT?
# While FunctionTransformer (like Log) requires YOU to know
# exactly what math to apply, PowerTransformer is DATA-DRIVEN.
#
# It looks at your column and AUTOMATICALLY finds the best
# mathematical transformation to make it look like a
# Normal Distribution (Bell Curve).
#
# KEY DIFFERENCE FROM FUNCTION TRANSFORMER:
# FunctionTransformer → YOU decide the math (log, sqrt etc.)
# PowerTransformer    → IT decides the math automatically
#                       by calculating λ (lambda) from data
# ============================================================


# ============================================================
# WHEN TO USE WHICH METHOD?
# ============================================================
#
# ┌──────────────┬────────────────────────────────────────┐
# │ Technique    │ When to use                            │
# ├──────────────┼────────────────────────────────────────┤
# │ Box-Cox      │ Only when ALL values are > 0 (positive)│
# │ Yeo-Johnson  │ Works for BOTH positive AND negative   │
# │              │ numbers → SAFER, use this by default!  │
# └──────────────┴────────────────────────────────────────┘
#
# REAL EXAMPLE:
# Titanic Fare → all positive → Box-Cox OR Yeo-Johnson ✅
# Temperature  → can be negative → ONLY Yeo-Johnson ✅
# Age          → all positive but use Yeo-Johnson anyway
#                (safer choice, handles edge cases better)


# ============================================================
# LOG vs POWER TRANSFORMER — KEY DIFFERENCE
# ============================================================
#
# LOG (np.log1p):
# → CONSTANT math → always applies the same curve
# → Predictable but might not perfectly normalize data
# → YOU choose the transformation
#
# POWER TRANSFORMER:
# → PARAMETRIC math → calculates λ (lambda) FROM your data
# → λ tells it exactly how much to "bend" the distribution
# → AUTOMATICALLY finds the best transformation
# → More likely to achieve true normal distribution
#
# ANALOGY:
# Log = using a fixed size wrench (might not fit perfectly)
# PowerTransformer = using an adjustable wrench that
#                    automatically adjusts to the perfect size!
#
# BONUS: PowerTransformer also STANDARDIZES automatically
# (mean=0, std=1) — you don't need a separate StandardScaler!


# ============================================================
# IMPORTS
# ============================================================

from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ============================================================
# BASIC USAGE
# ============================================================

# method='yeo-johnson' → safer default, handles any values
# method='box-cox'     → only use when all values > 0

transformer_power = ColumnTransformer([
    ('power_trf',
     PowerTransformer(method='yeo-johnson'),
     ['age', 'fare'])   # only apply to NUMERICAL columns!
],
remainder='passthrough',
verbose_feature_names_out=False)

# AND STANDARDIZATION ALSO OCCURS AUTOMATICALLY
# IN POWER TRANSFORMER — no need for separate StandardScaler!

# Add it to your pipeline sequence:
# pipe = Pipeline([
#     ('trf1', transformer1),      # imputation
#     ('trf2', transformer2),      # encoding
#     ('trf_power', transformer_power), # power transform + scale
#     ('model', LinearRegression())
# ])


# ============================================================
# KEY RULES — DO's AND DON'Ts
# ============================================================
#
# ✅ DO use on numerical features that look like a "slide"
#    (skewed distribution) instead of a "bell curve"
#
# ✅ DO use Yeo-Johnson by default (safer than Box-Cox)
#
# ✅ DO use inside a Pipeline (follows same golden rule:
#    fit_transform on train, transform on test only)
#
# ❌ DON'T use on categorical columns
#    (encoding numbers like 0,1,2 for categories → meaningless!)
#
# ❌ DON'T use on your target variable y
#    (unless specifically needed for regression output)
#
# ❌ DON'T use Box-Cox if data has zeros or negative values
#    → it will CRASH. Use Yeo-Johnson instead.


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. PowerTransformer = automatic version of FunctionTransformer
#    Finds the BEST math transformation for your data
#    using λ (lambda) calculated FROM your data
#
# 2. Two methods:
#    Yeo-Johnson → default, works for any values ✅
#    Box-Cox     → only for strictly positive values
#
# 3. Automatically STANDARDIZES as well (mean=0, std=1)
#    No need for separate StandardScaler after this!
#
# 4. Use it on NUMERICAL columns that are SKEWED
#    Check with sns.distplot() first
#    Skewed → use PowerTransformer
#    Already normal → skip it, not needed
#
# 5. Follows same golden rule as everything else:
#    fit_transform(X_train) → transform(X_test) only
# ============================================================


# ============================================================
# COMING UP NEXT:
# → ML Algorithms (Linear Regression, Logistic Regression)
#    Now that preprocessing is fully covered, we finally
#    start training actual ML models and understanding
#    how each algorithm learns from the data!
# ============================================================