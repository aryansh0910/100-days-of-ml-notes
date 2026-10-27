# ============================================================
# ADABOOST HYPERPARAMETERS
# ============================================================

# ============================================================
# WHAT IS IT?
#
# Hyperparameters in AdaBoost are slightly different from
# Random Forest because we aren't just building a forest;
# we are building a CHAIN OF CORRECTIONS. The goal is to
# balance how fast the model learns versus how much it
# "obsesses" over mistakes.
# ============================================================


# ============================================================
# 1. THE "BIG THREE" HYPERPARAMETERS
#
# ESTIMATOR (formerly base_estimator):
# This is the model that AdaBoost will try to improve.
# Default: A Decision Stump (DecisionTreeRegressor(max_depth=1))
#
# PRO TIP: You can use deeper trees (e.g. max_depth=3),
# but be careful. If the base learner is too STRONG,
# AdaBoost loses its "boosting" power and overfits quickly.
#
# VISUAL — base learner strength tradeoff:
#
# max_depth=1 (stump, weak):
# Each stump catches one mistake at a time
# → slow but stable, 100+ stumps needed
# → full boosting power ✅
#
# max_depth=3 (medium):
# Each tree catches more mistakes at once
# → faster learning but overfits with fewer trees ⚠️
#
# max_depth=10 (too strong):
# Each tree nearly memorizes the data
# → AdaBoost has nothing left to "correct" ❌
# → loses boosting advantage entirely
#
# ─────────────────────────────────────────────────────────
#
# N_ESTIMATORS: The number of models in the sequence.
#
# THE TRADEOFF: Unlike Random Forest, MORE IS NOT ALWAYS
# BETTER. If you have too many estimators, AdaBoost will
# eventually start "memorizing" noise and outliers, leading
# to OVERFITTING.
#
# VISUAL — Random Forest vs AdaBoost as n increases:
#
# Accuracy
#  |         Random Forest:
#  |              ●●●●●●●●●●●  (plateaus, never drops)
#  |         AdaBoost:
#  |           ●●●●●●
#  |         ●●        ●●●      (peaks then DROPS if too many)
#  |       ●●              ●●   ← overfitting to noise
#  |___________________________________ n_estimators
#
# ─────────────────────────────────────────────────────────
#
# LEARNING_RATE: How much "weight" we give to the
# corrections of each new tree.
#
# DIRECT RELATIONSHIP: If you DECREASE the learning_rate,
# you usually need to INCREASE n_estimators to give the
# model enough time to learn.
# ============================================================


# ============================================================
# 2. THE RELATIONSHIP — LEARNING RATE vs ESTIMATORS
#
# Think of it like STEERING A CAR:
#
# HIGH Learning Rate (1.0): Sharp turns. You reach your
# destination fast, but you might oversteer and CRASH
# (overfit).
#
# LOW Learning Rate (0.1): Gentle turns. It takes longer
# to get there, but the path is much smoother and more
# stable.
#
# VISUAL:
#
# High LR (1.0), few estimators:
# Start ──────────────────→ ✅ (fast but risky)
#
# Low LR (0.1), many estimators:
# Start ──→──→──→──→──→──→──→──→──→ ✅ (slow but stable)
#
# Low LR (0.1), too FEW estimators:
# Start ──→──→──→  (never reaches the answer) ❌
#
# RULE OF THUMB:
# learning_rate=0.1 → n_estimators=500
# learning_rate=1.0 → n_estimators=50-100
# They compensate for each other — tune them TOGETHER.
# ============================================================


# ============================================================
# 4. THE loss PARAMETER (SPECIFIC TO REGRESSORS)
#
# In CLASSIFICATION, AdaBoost uses "errors."
# In REGRESSION, it uses a LOSS FUNCTION to decide how to
# update weights:
#
# LINEAR (Default): Weights updated based on DIRECT error.
# Good starting point for most problems.
#
# SQUARE: Errors are SQUARED. This penalizes large errors
# much more heavily. Good for catching big misses.
#
# EXPONENTIAL: Updates weights EXPONENTIALLY. The most
# aggressive setting — large errors get massive weight
# increases.
#
# VISUAL — same error, different penalty:
#
# Error = 5 (a moderately wrong prediction):
# Linear      → penalty = 5
# Square      → penalty = 25   ← much harsher
# Exponential → penalty = e^5 ≈ 148 ← extremely harsh
#
# Use Square or Exponential when you REALLY can't afford
# large prediction errors (e.g. medical dosage prediction).
# ============================================================


# ============================================================
# 5. WHEN TO STOP — THE "OVERFITTING" CHECK
#
# Because AdaBoost is so aggressive, you should ALWAYS
# check the difference between Training Score and Test Score.
#
# If Train = 0.95 and Test = 0.60 → you have OVERFITTED.
#
# FIX:
# → Reduce n_estimators
# → Reduce learning_rate
# → Reduce max_depth of the base tree
#
# VISUAL — diagnosing AdaBoost overfitting:
#
# Score
#  1.0 |         Train: ●●●●●●●●●●●●●● (keeps climbing)
#  0.8 |         Test:  ●●●●●●
#  0.6 |                       ●●●●     ← test DROPS here
#  0.4 |                           ●●   ← overfitting zone
#  |_________________________________ n_estimators
#  0   50  100  150  200  250  300
#
# Pick n_estimators at the point where the TEST score
# is highest — before it starts dropping.
# ============================================================


# ============================================================
# CODE — GRIDSEARCHCV FOR ADABOOST
# ============================================================

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# 1. Setup the search space
# We test different depths and different learning speeds
param_grid = {
    'estimator'    : [DecisionTreeRegressor(max_depth=1),
                      DecisionTreeRegressor(max_depth=3)],
    'n_estimators' : [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0],
    'loss'         : ['linear', 'square', 'exponential']
    # How to calculate the error for weight updates
}

# 2. Run the Grid Search
# 2×3×3×3 = 54 combinations × 5 folds = 270 fits
ada_search = GridSearchCV(
    AdaBoostRegressor(random_state=42),
    param_grid,
    cv=5,
    n_jobs=-1   # use all CPU cores
)
ada_search.fit(X_reg, y_reg)

print(f"Best Score:  {ada_search.best_score_:.4f}")
print(f"Best Params: {ada_search.best_params_}")


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Big Three: estimator, n_estimators, learning_rate
#
# 2. Base estimator should be WEAK (max_depth=1 or 2).
#    Too strong → AdaBoost loses boosting power, overfits.
#
# 3. Unlike Random Forest, more n_estimators is NOT always
#    better — AdaBoost can overfit if given too many stumps.
#    Always check train vs test score.
#
# 4. learning_rate and n_estimators work TOGETHER:
#    Low LR → need more estimators (compensate each other)
#    High LR → need fewer estimators
#
# 5. loss (regressor only):
#    linear → default, safe
#    square → punishes big errors more
#    exponential → most aggressive
#
# 6. OVERFITTING CHECK: if train >> test score → reduce
#    n_estimators, reduce learning_rate, or reduce depth
#
# 7. INTERVIEW TRAP:
#    Q: "Does adding more estimators always help AdaBoost
#        like it does in Random Forest?"
#    Right answer: "No — this is a key difference. In
#    Random Forest, more trees just add stability since
#    they're independent. In AdaBoost, trees are sequential
#    and each one focuses more on difficult/noisy points.
#    After a certain point, AdaBoost starts 'memorizing'
#    outliers that got high weights, causing the test score
#    to actually DROP while training score keeps climbing.
#    Always monitor the train-test gap and stop early."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Gradient Boosting
#    More powerful than AdaBoost — instead of updating
#    sample weights, it fits each new tree directly on the
#    RESIDUAL ERRORS (leftover mistakes) of the previous
#    model.
# ============================================================