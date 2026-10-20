# ============================================================
# LASSO REGRESSION (L1)
# ============================================================

# ============================================================
# WHAT IS IT?
#
# CHANCES OF OVERFITTING IS VERY HIGH IN CASE OF HIGH
# DIMENSIONAL DATA, I.E. WHERE THERE ARE MANY FEATURES. SO
# WE JUST SELECT SOME BEST COLUMNS AND REMOVE OTHERS TO
# PERFORM THE REGRESSION.
#
# IF U USE RIDGE REGRESSION, THE VALUES OF EACH COLUMN WILL
# DECREASE FOR SURE BUT THE COLUMN WILL STILL BE INTACT,
# WHICH IS NOT THE CASE IN LASSO REGRESSION, BECAUSE IT
# MAKES THE WHOLE COLUMN 0, I.E. THE LESS IMPORTANT COLUMN
# DOESN'T MATTER AT ALL THEN.
#
# If Ridge is the "gentle stabilizer," Lasso (Least
# Absolute Shrinkage and Selection Operator) is the
# "ruthless minimalist."
#
# While Ridge shrinks coefficients closer to zero, Lasso
# has a unique geometric property that allows it to set
# coefficients to EXACTLY zero. This turns Lasso into a
# built-in Feature Selection tool.
# ============================================================


# ============================================================
# 1. THE MATH — L1 REGULARIZATION
#
# Lasso changes the cost function by adding a penalty
# based on the ABSOLUTE VALUE of the weights
# (vs Ridge, which uses the SQUARE of the weights).
# ============================================================


# ============================================================
# 2. GEOMETRIC INTUITION — THE DIAMOND vs THE CIRCLE
#
# This is the MOST IMPORTANT part of understanding Lasso.
#
# RIDGE (L2) has a CIRCULAR constraint. The error contours
# usually hit the circle at a point where BOTH weights are
# still alive (small, but non-zero).
#
# LASSO (L1) has a DIAMOND-shaped constraint. Because the
# diamond has "sharp corners" sitting exactly ON the axes,
# the error contours are very likely to hit a corner FIRST.
#
# VISUAL — why the shape matters:
#
# RIDGE (circle constraint):
#         w2
#          |
#       ___|___
#      /   |   \      ← error contour touches the
#     |    ●----│        SMOOTH circle edge at a
#      \___|___/         point where BOTH w1, w2 > 0
#          |________ w1
#    Both weights survive, just smaller
#
# LASSO (diamond constraint):
#         w2
#          |
#         /|\
#        / | \
#       ●--+--●         ← error contour touches a
#        \ | /             SHARP CORNER where one
#         \|/              axis value = 0
#          |________ w1
#    One weight gets killed to exactly 0!
#
# The sharp corners of the diamond sit exactly on the
# axes (where one coefficient = 0). Geometrically, the
# contour lines are much more likely to intersect at
# these corners than anywhere else — hence Lasso naturally
# produces zero coefficients.
# ============================================================


# ============================================================
# 3. LASSO'S "SUPERPOWER" — FEATURE SELECTION
#
# Imagine your Churn dataset has 50 columns after One-Hot
# Encoding. Some might be useless (like a specific small
# city or a redundant category).
#
# RIDGE will give those useless columns TINY weights
# (e.g. 0.0001). You still have to manage 50 columns.
#
# LASSO will give them a weight of 0.0. It effectively
# "DELETES" them. You might end up with only 10 important
# features, making your model much easier to explain to
# your boss.
#
# VISUAL:
#
# Ridge output (50 features, all alive):
# City_A: 0.3, City_B: 0.0001, City_C: 0.00003, ... (50 total)
#
# Lasso output (50 features, sparse):
# City_A: 0.3, City_B: 0.0, City_C: 0.0, ... (only ~10 non-zero)
# ============================================================


# ============================================================
# 4. THE WEAKNESS OF LASSO
#
# Lasso isn't always perfect. It has one major flaw
# compared to Ridge:
#
# THE CORRELATION PROBLEM:
# If you have two features that are highly correlated
# (e.g. Monthly_Usage and Total_Bill), Lasso will likely
# pick ONE at random, give it a weight, and set the OTHER
# to zero.
#
# WHY THIS IS BAD:
# You might lose a feature that was actually important for
# understanding the business, just because it was similar
# to another one.
#
# VISUAL:
#
# Two correlated features going into Lasso:
# Monthly_Usage  → weight = 0.6  ← kept (randomly chosen)
# Total_Bill     → weight = 0.0  ← killed, even though it
#                                   might have been equally
#                                   or more meaningful!
# ============================================================


# ============================================================
# CODE
# ============================================================

from sklearn.linear_model import Lasso
import pandas as pd

# alpha=1.0 is the penalty. Higher alpha = fewer features survive.
lasso = Lasso(alpha=0.1)


# ============================================================
# ALPHA IN LASSO — THE FILTER
#
# In Lasso, Alpha acts as a FILTER.
#
# BEHAVIOR: As you increase Alpha, Lasso starts "killing"
# features. It sets their coefficients to exactly 0.0.
#
# SPARSITY: High Alpha in Lasso creates a "Sparse Model" —
# a model with very few non-zero weights.
#
# THE "SELECTION" RULE: At a high enough Alpha, only the
# most important features survive.
#
# BEST FOR: When you have 100 features but suspect only 5
# are actually important.
# ============================================================

# Fit it (Scaling is MANDATORY here too!)
lasso.fit(X_train_scaled, y_train)

# Check which features were "killed"
coef_df = pd.DataFrame({'Feature': features, 'Coef': lasso.coef_})
print(coef_df[coef_df['Coef'] == 0])  # These are the columns Lasso dropped!


# ============================================================
# RIDGE vs LASSO — WHEN TO USE WHICH
#
# USE RIDGE in case where U want to keep all the columns
# and can't drop any column.
#
# USE LASSO where U can delete some columns — it's like
# a feature selection technique.
# ============================================================


# ============================================================
# RIDGE vs LASSO — FULL COMPARISON
#
# ┌──────────────────┬──────────────────────┬───────────────────────┐
# │                  │ Ridge (L2)           │ Lasso (L1)            │
# ├──────────────────┼──────────────────────┼───────────────────────┤
# │ Penalty term     │ Sum of squared       │ Sum of absolute       │
# │                  │ weights              │ weights               │
# │ Constraint shape │ Circle               │ Diamond               │
# │ Coefficients     │ Shrink, never zero   │ Shrink, CAN become    │
# │                  │                      │ exactly zero          │
# │ Feature selection│ No                   │ Yes (built-in)        │
# │ Correlated       │ Keeps both, shares   │ Picks one randomly,   │
# │ features         │ the weight           │ zeroes the other      │
# │ Best for         │ Many features, all   │ Many features, only   │
# │                  │ somewhat useful      │ a few actually matter │
# └──────────────────┴──────────────────────┴───────────────────────┘
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Lasso penalty uses ABSOLUTE value of weights (vs
#    Ridge's SQUARED value)
#
# 2. THE GEOMETRIC REASON Lasso zeros coefficients: diamond
#    shaped constraint has sharp corners ON the axes —
#    error contours hit corners more often than circle edges
#
# 3. Lasso = automatic feature selection. Useless features
#    get weight = 0.0 exactly, effectively removed
#
# 4. WEAKNESS: with correlated features, Lasso randomly
#    picks one to keep and kills the other — might lose a
#    genuinely useful feature
#
# 5. Alpha controls sparsity: higher alpha = more features
#    killed = sparser model
#
# 6. Use Ridge when you want to keep ALL features (just
#    dampened). Use Lasso when you suspect many features
#    are genuinely useless and want them removed.
#
# 7. ALWAYS scale data before Lasso — same reason as Ridge,
#    penalty is unfair across unscaled feature ranges
#
# 8. INTERVIEW TRAP:
#    Q: "Why does Lasso produce exactly zero coefficients
#        while Ridge doesn't?"
#    Right answer: "It's a geometric property. Lasso's L1
#    constraint region is a diamond with sharp corners
#    sitting exactly on the axes — error contour lines are
#    much more likely to intersect at a corner, which means
#    one weight = 0. Ridge's L2 constraint is a smooth
#    circle with no corners, so contours rarely land exactly
#    on an axis — weights shrink but stay non-zero."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → ElasticNet Regression
#    Combining L1 and L2 penalties to get the best of both
#    Ridge and Lasso in one model.
# ============================================================