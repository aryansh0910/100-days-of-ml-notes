# ============================================================
# DECISION TREE REGRESSOR
# ============================================================

# ============================================================
# WHAT IS IT?
#
# When we switch from Classification Trees to Regression
# Trees, the goal changes from predicting a "Category"
# (Yes/No) to predicting a CONTINUOUS NUMBER (like Price,
# Temperature, or Salary).
#
# The geometric logic remains the same (cutting the scatter
# plot into boxes), but how the "winner" is decided inside
# those boxes is different.
# ============================================================


# ============================================================
# 1. THE CORE DIFFERENCE
#
# CLASSIFICATION TREES: The prediction for a leaf node is
# the MODE (the most frequent class in that box).
#
# REGRESSION TREES: The prediction for a leaf node is the
# MEAN (the average value of all points in that box).
#
# VISUAL:
#
# Classification leaf:
# Box contains: [Churn, Stay, Churn, Churn]
# Prediction = MODE = Churn (appears 3 times)
#
# Regression leaf:
# Box contains: [150k, 170k, 140k, 160k]
# Prediction = MEAN = (150+170+140+160)/4 = 155k
# Every new house that falls in this box gets 155k ✅
# ============================================================


# ============================================================
# 2. HOW IT "DECIDES" WHERE TO SPLIT
#
# In Classification, we used Gini or Entropy to measure
# "messiness." In Regression, we use MSE (Mean Squared
# Error) or MAE (Mean Absolute Error).
#
# The tree tries various split points and chooses the one
# that MINIMIZES THE VARIANCE within the resulting boxes.
# It wants the numbers in each box to be as SIMILAR to
# each other as possible.
#
# VISUAL — what a good split looks like:
#
# BAD SPLIT (high variance in each box):
# Box 1: [150k, 400k, 200k, 500k]  ← wildly different ❌
# Box 2: [180k, 450k, 160k, 480k]  ← wildly different ❌
#
# GOOD SPLIT (low variance in each box):
# Box 1: [140k, 150k, 160k, 155k]  ← all similar ✅
# Box 2: [480k, 500k, 490k, 510k]  ← all similar ✅
#
# The tree picks the split that creates the most
# "internally consistent" boxes — same logic as
# Gini/Entropy, just measured with MSE instead.
# ============================================================


# ============================================================
# 3. GEOMETRIC MAP — PREDICTING HOUSE PRICE
#
# Instead of a smooth line, the tree creates a "STAIRCASE"
# of averages.
#
# PRICE ($)
#  ^
#  |          Box 3 Average: 500k
#  |                     +---------
#  |          Box 2 Avg: |
#  |          300k  +----|
#  |   Box 1 Avg:   |
#  |   150k  +------|
#  |         |      |
#  0 +-------+------+-------------+------> SQ FT
#    0      1000   2000          3000
#
# WHY "STAIRCASE" AND NOT A SMOOTH LINE?
# Because each box gives ONE average prediction — so the
# output is flat within each box and JUMPS at every split
# boundary. More depth = more boxes = more stairs =
# smoother approximation of the real curve.
# ============================================================


# ============================================================
# 4. RULE-BASED EXAMPLE
#
# If you were to look at the rules for the map above:
#
# IF Sq_Ft < 1000:               Then Price = 150,000
# IF Sq_Ft between 1000 and 2000: Then Price = 300,000
# IF Sq_Ft > 2000:               Then Price = 500,000
#
# Notice that every house between 1000 and 2000 sq ft gets
# the EXACT SAME prediction ($300k). To get more precise,
# the tree needs more depth — more splits = more boxes =
# more granular predictions.
#
# THIS IS THE CORE LIMITATION OF REGRESSION TREES vs
# LINEAR REGRESSION:
# Linear Regression → smooth, continuous line of predictions
# Regression Tree   → staircase, same value per box
#
# For very smooth relationships, Linear Regression is
# better. For complex non-linear patterns, the tree wins.
# ============================================================


# ============================================================
# CODE
# ============================================================

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# 1. Initialize
# max_depth is still very important to prevent overfitting
reg_tree = DecisionTreeRegressor(max_depth=3,
                                  criterion='squared_error')

# 2. Fit (No scaling needed — same reason as classifier)
reg_tree.fit(X_train, y_train)

# 3. Predict (Returns continuous numbers, not classes)
y_pred = reg_tree.predict(X_test)

# 4. Evaluate using RMSE
mse  = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
print("Average error in prediction:", rmse)
# RMSE is in the SAME UNITS as your target (dollars,
# degrees, salary) — much more interpretable than raw MSE


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Regression Tree = same structure as Classification Tree
#    ONLY difference: leaf prediction is MEAN not MODE
#
# 2. Split criterion changes:
#    Classification → Gini / Entropy (measures class purity)
#    Regression     → MSE / MAE (measures value variance)
#
# 3. Output is a STAIRCASE not a smooth line — every point
#    in the same leaf gets the EXACT same predicted value
#
# 4. More depth = more boxes = finer staircase =
#    better approximation, but higher overfitting risk
#
# 5. No scaling needed — same reason as classification tree
#    (splits based on thresholds, not distances)
#
# 6. Evaluate with RMSE/MAE, not accuracy — it's a
#    regression problem, not classification
#
# 7. INTERVIEW TRAP:
#    Q: "How does a Regression Tree make predictions
#        differently from a Classification Tree?"
#    Right answer: "Both use the same splitting logic —
#    finding the threshold that creates the purest groups.
#    But Classification uses Gini/Entropy to measure purity
#    and predicts the MODE of each leaf. Regression uses
#    MSE to measure purity (minimizing variance within each
#    box) and predicts the MEAN of each leaf. The output is
#    a staircase of averages, not a smooth line."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Random Forest
#    Combining many Decision Trees into an ensemble to
#    reduce overfitting and improve the staircase
#    approximation significantly.
# ============================================================