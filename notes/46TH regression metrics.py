# ============================================================
# REGRESSION METRICS — MAE, MSE, RMSE, R²
# ============================================================

# ============================================================
# WHAT IS IT?
#
# Regression metrics tell us about how good our model is.
# ============================================================


# ============================================================
# 1. MAE (MEAN ABSOLUTE ERROR)
#
# This is the most intuitive one. It simply calculates the
# average of the absolute differences between the predicted
# values (ŷ) and the actual values (y).
#
# VIBE: "On average, my prediction is off by X units."
#
# PRO: It's in the SAME UNITS as your target
#      (e.g. if you're predicting income, MAE is in Dollars)
#
# CON: It treats small errors and huge errors with the SAME
#      "severity" — a 1000-error and a 10-error of equal
#      total count both count proportionally the same.
#
# FORMULA:
# ┌─────────────────────────────────────────────┐
# │              1                              │
# │   MAE  =  ───  Σ |y_i - ŷ_i|                │
# │              n                              │
# └─────────────────────────────────────────────┘
# ============================================================


# ============================================================
# 2. MSE (MEAN SQUARED ERROR)
#
# This is the "strict teacher" of metrics. It squares the
# errors before averaging them.
#
# VIBE: It punishes large errors (outliers) heavily.
#
# PRO: Useful if you want to ensure your model never makes
#      a "huge" mistake.
#
# CON: The units are squared (e.g. "Dollars Squared"),
#      which makes no sense to humans.
#
# FORMULA:
# ┌─────────────────────────────────────────────┐
# │              1                              │
# │   MSE  =  ───  Σ (y_i - ŷ_i)²               │
# │              n                              │
# └─────────────────────────────────────────────┘
#
# VISUAL — why MSE punishes harder:
#
# Error of 2  → MAE contributes 2,  MSE contributes 4
# Error of 10 → MAE contributes 10, MSE contributes 100
#
# Notice: MAE grows linearly, MSE grows QUADRATICALLY.
# A single huge mistake dominates the MSE total.
# ============================================================


# ============================================================
# 3. RMSE (ROOT MEAN SQUARED ERROR)
#
# This is the most popular metric for reporting results.
# It's just the square root of the MSE.
#
# VIBE: Gives you the "punishing" nature of MSE but brings
# the units back to something we can understand (like
# Dollars).
#
# WHY USE IT?
# If your RMSE is much higher than your MAE, it's a RED
# FLAG that you have outliers in your data that are
# confusing the model.
#
# FORMULA:
# ┌─────────────────────────────────────────────┐
# │   RMSE  =  √MSE                              │
# └─────────────────────────────────────────────┘
#
# VISUAL — reading the gap between MAE and RMSE:
#
# Case A: MAE = 50,  RMSE = 55   → close, errors are even,
#                                   no major outliers ✅
#
# Case B: MAE = 50,  RMSE = 200  → BIG gap, means a few
#                                   predictions are WILDLY
#                                   off — outliers present ⚠️
# ============================================================


# ============================================================
# 4. R² SCORE (COEFFICIENT OF DETERMINATION)
#
# While the others tell you "How much error do I have?",
# R² tells you "How much BETTER is my model than a dumb
# guess?"
#
# A "dumb guess" is just predicting the MEAN of the data
# for everyone.
#
# RANGE: Usually 0 to 1.
# 0 → Your model is as bad as just guessing the average
# 1 → Perfect model (zero error)
#
# PRO: It's independent of units, so you can compare an
# "Income Model" to a "CGPA Model."
#
# VISUAL — what R² actually compares:
#
# Dumb guess (predict mean for everyone):
#  y |  ────────────●───────●──────  ← flat line at mean
#    |        ●            ●
#    |     ●        ●
#
# Your model (actual predictions):
#  y |        ╱●          ●
#    |     ●╱      ●  ╱●
#    |  ●╱      ●  ╱
#
# R² measures: how much closer does YOUR line get to the
# real points compared to that flat "dumb guess" line?
# ============================================================


# ============================================================
# CODE
# ============================================================

import numpy as np
from sklearn.metrics import (mean_absolute_error,
                              mean_squared_error,
                              r2_score)

y_pred = lr.predict(X_test)

print("MAE", mean_absolute_error(y_test, y_pred))
print("MSE", mean_squared_error(y_test, y_pred))
print("RMSE", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2", r2_score(y_test, y_pred))


# ============================================================
# QUICK COMPARISON TABLE
#
# ┌─────────┬──────────────────┬──────────────┬─────────────┐
# │ Metric  │ Units            │ Outlier      │ Best for    │
# │         │                  │ sensitivity  │             │
# ├─────────┼──────────────────┼──────────────┼─────────────┤
# │ MAE     │ Same as target   │ Low          │ Robust view │
# │ MSE     │ Squared (weird)  │ High         │ Math/optim. │
# │ RMSE    │ Same as target   │ High         │ Reporting   │
# │ R²      │ Unitless (0-1)   │ Moderate     │ Comparing   │
# │         │                  │              │ models      │
# └─────────┴──────────────────┴──────────────┴─────────────┘
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. MAE → average absolute error, same units, treats all
#    errors equally
#
# 2. MSE → squares errors, punishes big mistakes harder,
#    but units become meaningless (squared)
#
# 3. RMSE → square root of MSE, brings units back, MOST
#    commonly reported metric
#
# 4. Big gap between MAE and RMSE → red flag for outliers
#
# 5. R² → compares your model to a "dumb" baseline (predicting
#    the mean). 0 = useless, 1 = perfect. Unitless, so you
#    can compare across different problems.
#
# 6. INTERVIEW TRAP:
#    Q: "Which metric should you always use?"
#    Wrong: "RMSE, it's the most popular"
#    Right: "Depends — RMSE for general reporting, but
#    if you suspect outliers are present, ALSO check MAE.
#    A large MAE-RMSE gap reveals outlier influence that
#    RMSE alone would hide. R² is useful when comparing
#    across different models/datasets since it's unitless."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Logistic Regression
#    Adapting the linear equation with a Sigmoid function
#    to handle binary classification problems like Churn.
# ============================================================