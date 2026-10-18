# ============================================================
# REGULARIZATION — RIDGE REGRESSION (L2)
# ============================================================

# ============================================================
# WHAT IS IT?
#
# Regularization is a technique used to prevent a machine
# learning model from overfitting.
#
# It's the "discipline" we apply to a model when it starts
# getting too obsessed with the noise and tiny details of
# the training data.
# ============================================================


# ============================================================
# 1. THE CORE CONCEPT — THE COMPLEXITY PENALTY
#
# When a model overfits, it usually develops very large
# coefficients (weights). It thinks, "To predict Churn
# perfectly, I need to multiply Income by 5,000 and
# subtract Age by 2,000!" These massive numbers make the
# model fragile.
#
# Regularization changes the GOAL of the model:
#
# STANDARD GOAL:     Minimize Error (Loss)
# REGULARIZED GOAL:  Minimize (Error + Penalty)
#
# The Penalty is based on the SIZE of the weights. If the
# model wants to use a huge weight, it has to "pay" for it
# in the cost function. This forces the model to stay simple.
#
# VISUAL — what "fragile" means:
#
# Overfit model (huge weights):
# Predict = 5000×Income - 2000×Age + ...
# A tiny change in Income (say $1) swings the prediction
# wildly because of the massive multiplier ❌
#
# Regularized model (small weights):
# Predict = 0.5×Income - 0.2×Age + ...
# A tiny change in Income barely moves the prediction —
# stable and reliable ✅
# ============================================================


# ============================================================
# OVERFITTING vs UNDERFITTING — IN TERMS OF SLOPE (M)
#
# OVERFITTING IN TERMS OF LINEAR REGRESSION IS LIKE THE
# VALUE OF M IN y = mx + b BEING VERY HIGH — MEANS IT HAS
# A LOT OF WEIGHT.
#
# BUT WHEN THE VALUE OF M IS VERY SMALL, SO SMALL THAT IT
# HAS NO ROLE IN IT, THEN THAT IS KNOWN AS UNDERFITTING.
#
# SO REGULARIZATION IS DECREASING THE VALUE OF M IN THE
# CASE OF OVERFITTING.
#
# VISUAL:
#
# Overfitting (m too large):
#  y |     ●
#    |  ● ╱│●          ← line bends wildly to hit every point
#    | ╱  │ ╲  ●
#    |╱   │  ╲   ●
#    |____│___╲____________ x
#
# Underfitting (m too small, nearly flat):
#  y |  ●     ●    ●     ●
#    |___________________  ← line barely responds to x at all
#    |              x
#
# Good fit (m just right — regularization's goal):
#  y |     ●
#    |  ●    ╱     ●
#    |    ╱    ●
#    |  ╱____________________ x
# ============================================================


# ============================================================
# 3. THE BIAS-VARIANCE TRADE-OFF
#
# Regularization intentionally adds a little bit of BIAS
# (making the model slightly less accurate on the training
# data) in exchange for a MASSIVE reduction in VARIANCE
# (making it much more accurate on new, unseen data).
#
# VISUAL:
#
# No regularization:
# Train accuracy: 99%  ← very low bias
# Test accuracy:  60%  ← very high variance (overfit!)
#
# With regularization:
# Train accuracy: 92%  ← slightly more bias (small sacrifice)
# Test accuracy:  88%  ← much lower variance (generalizes!)
# ============================================================


# ============================================================
# THE TWO MAIN TYPES
#
# L2 REGULARIZATION (RIDGE):
# Adds a penalty proportional to the SQUARE of the weights.
# Shrinks all weights but keeps them all alive.
#
# L1 REGULARIZATION (LASSO):
# Adds a penalty proportional to the ABSOLUTE VALUE of the
# weights. It is "ruthless" and can turn useless weights to
# EXACTLY ZERO, effectively deleting those features.
# ============================================================


# ============================================================
# RIDGE REGRESSION — IN DEPTH
#
# Welcome to the world of Regularization! If Linear
# Regression is a student who tries to memorize every
# single detail of the textbook (leading to overfitting),
# Ridge Regression is the teacher who tells them, "Focus on
# the big picture and don't get obsessed with the tiny
# details."
#
# THE PROBLEM — OVERFITTING & LARGE COEFFICIENTS:
# In Multiple Linear Regression, when you have many
# features (like in your churn data) or high correlation
# between features, the model often tries to be too
# "perfect." It assigns massive weights (β) to features to
# catch every tiny wobble in the data. This makes the model
# fragile. A tiny change in a new customer's data could
# result in a wild, incorrect prediction.
#
# THE SOLUTION — L2 REGULARIZATION:
# Ridge Regression adds a penalty to the Cost Function (MSE).
# It tells the model: "I want you to minimize the error,
# BUT I also want you to keep the weights (w) as small as
# possible."
# ============================================================


# ============================================================
# ALPHA — THE PENALTY STRENGTH
#
# α = 0:
# You have ZERO Regularization. The model is just standard
# Linear Regression. It will try to hit every data point
# perfectly, risking Overfitting.
#
# α → ∞ (Infinity):
# The penalty becomes so massive that the model is forced
# to make the coefficients as small as possible. The model
# becomes a FLAT LINE (underfitting).
#
# THE "SWEET SPOT":
# A middle-ground Alpha that reduces noise but keeps the
# actual signal.
#
# VISUAL — the alpha dial:
#
# α=0          α=1 (sweet spot)        α=1000
# Overfitting  Good Fit                Underfitting
#  ●  ╱│╲ ●         ●  ╱                  ────────
#  ╱  │ ╲           ╱  ●                  (flat line,
# ╱   │  ●        ╱     ●                  no signal)
# (wiggly line    (smooth, balanced
#  hits every       fit)
#  point)
# ============================================================


# ============================================================
# CODE
# ============================================================

from sklearn.linear_model import Ridge
import pandas as pd

# 1. Initialize (alpha is your 'lambda' penalty strength)
# alpha=0 is same as Linear Regression. Higher alpha = more shrinkage.
ridge = Ridge(alpha=1.0)

# In Ridge, Alpha controls how much we squeeze the
# coefficients toward zero.
#
# BEHAVIOR: As you increase Alpha, the coefficients w get
# smaller and smaller (e.g. 5.0 → 0.5 → 0.01).
#
# THE "NEVER ZERO" RULE:
# No matter how high you turn the Alpha knob in Ridge, the
# weights will NEVER actually reach zero. They just become
# infinitesimally small.
#
# BEST FOR: When you have many features that all contribute
# a little bit, and you want to keep them ALL in the model.

# 2. Fit the model (Assumes data is already scaled!)
ridge.fit(X_train_scaled, y_train)

# 3. Predict
y_pred = ridge.predict(X_test_scaled)

print("Ridge coefficients:", ridge.coef_)
# Notice: small but non-zero values, even for weak features


# ============================================================
# WHY MUST DATA BE SCALED BEFORE RIDGE?
#
# Ridge penalizes the SIZE of weights. If Salary (range
# 20,000-150,000) and Age (range 18-80) are unscaled, the
# weight for Salary will naturally be tiny just because of
# its huge range, and Age's weight will be naturally large.
# Ridge would then unfairly penalize Age's weight more than
# Salary's, even if Age is equally or more important.
# Scaling first ensures the penalty is FAIR across features.
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Regularization = Minimize (Error + Penalty)
#    Forces the model to keep weights small → simpler,
#    more generalizable model
#
# 2. Overfitting → weight (m) too large
#    Underfitting → weight (m) too small/near zero
#    Good regularization finds the sweet spot in between
#
# 3. Bias-Variance trade-off: regularization trades a small
#    increase in bias for a large decrease in variance
#
# 4. L2 (Ridge) → squares the weights in the penalty,
#    shrinks all weights but NEVER makes them exactly zero
#
# 5. L1 (Lasso) → uses absolute value in the penalty,
#    CAN shrink weights to exactly zero (feature selection)
#
# 6. Alpha controls penalty strength:
#    α=0 → plain Linear Regression (overfitting risk)
#    α=∞ → flat line (underfitting)
#    Sweet spot α → best generalization
#
# 7. ALWAYS scale data before Ridge/Lasso — unscaled
#    features get unfairly penalized based on their range,
#    not their actual importance
#
# 8. INTERVIEW TRAP:
#    Q: "Does Ridge Regression perform feature selection?"
#    Wrong: "Yes, it removes unimportant features"
#    Right: "No — Ridge shrinks ALL weights toward zero but
#    never makes them exactly zero, so all features remain
#    in the model. Lasso (L1) is the one that can zero out
#    weights and perform actual feature selection."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Lasso Regression (L1)
#    The "ruthless" regularization that can eliminate
#    useless features entirely by shrinking their weights
#    to exactly zero.
# ============================================================