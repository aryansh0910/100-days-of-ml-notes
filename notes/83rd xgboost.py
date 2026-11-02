# ============================================================
# XGBOOST — EXTREME GRADIENT BOOSTING
# ============================================================

# ============================================================
# WHAT IS IT?
#
# You've reached the "Final Boss" of tabular machine
# learning. XGBoost (Extreme Gradient Boosting) is
# essentially Gradient Boosting on steroids. It is famous
# for winning more Kaggle competitions than almost any
# other algorithm.
#
# Think of it as the elite version of the sculptor we
# talked about. While regular Gradient Boosting uses a
# CHISEL, XGBoost uses a HIGH-PRECISION LASER with
# built-in error correction.
#
# XGBOOST INBUILT HAS THE BINNING THING SO U SHOULDN'T
# SCALE YOUR DATA FOR XGBOOST. IT HANDLES THAT
# AUTOMATICALLY. IN FACT, SCALING CAN SOMETIMES HURT
# PERFORMANCE BECAUSE IT REMOVES THE NATURAL DISTRIBUTION
# OF THE DATA THAT XGBOOST RELIES ON FOR ITS SPLITS.
# ============================================================


# ============================================================
# 1. WHAT MAKES IT "EXTREME"?
#
# XGBoost is better than standard Gradient Boosting for
# THREE MAIN REASONS:
#
# A. REGULARIZATION (The L1/L2 Secret):
# Standard GBMs often overfit. XGBoost has "Lasso" and
# "Ridge" math built DIRECTLY into its objective function.
# This punishes complex models, forcing them to stay simple
# and generalize better.
#
# B. HARDWARE OPTIMIZATION:
# It's designed to use ALL the cores of your CPU in
# PARALLEL. Significantly faster than the standard
# Scikit-learn GBM version.
#
# C. HANDLING MISSING VALUES:
# XGBoost has a built-in "DEFAULT DIRECTION." If it
# encounters a missing value, it automatically LEARNS
# whether to send it to the left or right branch to get
# the best result. No manual imputation needed!
#
# VISUAL — GBM vs XGBoost:
#
# Standard GBM:                XGBoost:
# ┌─────────────────┐          ┌─────────────────────────┐
# │ Build tree      │          │ Build tree + Regularize │
# │ One core only   │          │ All CPU cores parallel  │
# │ Handle NaN: ❌  │          │ Handle NaN: auto ✅     │
# │ Prune: top-down │          │ Prune: bottom-up        │
# └─────────────────┘          └─────────────────────────┘
# ============================================================


# ============================================================
# 2. THE LOGIC — SIMILARITY SCORES AND GAIN
#
# Instead of just looking at "Gini" or "Entropy," XGBoost
# uses a SIMILARITY SCORE.
#
# 1. Calculates a SCORE for a group of residuals
# 2. Splits the data
# 3. Calculates the GAIN:
#    Gain = Similarity(Left) + Similarity(Right)
#           - Similarity(Parent)
# 4. If the Gain isn't HIGH ENOUGH, it PRUNES the branch
#    (deletes it), even if that branch seemed helpful
#    earlier.
#
# VISUAL — Gain calculation:
#
# Parent node (residuals: [+10, -5, +8, -3]):
# Similarity = (10-5+8-3)² / 4 = 100/4 = 25
#
# After split:
# Left  [+10, +8]  → Similarity = (18)² / 2 = 162
# Right [-5, -3]   → Similarity = (-8)² / 2 = 32
#
# Gain = 162 + 32 - 25 = 169 ← large gain, keep split ✅
#
# If Gain < gamma → PRUNE this split (remove it entirely)
# This is why XGBoost is more conservative than GBM!
# ============================================================


# ============================================================
# 4. HYPERPARAMETERS YOU MUST KNOW
#
# LEARNING_RATE (eta):
# Smaller is usually better, but you'll need more
# n_estimators. Controls how much each tree contributes.
#
# GAMMA:
# A node is split ONLY when the resulting split gives a
# POSITIVE REDUCTION in the loss function greater than
# gamma. Higher gamma = MORE CONSERVATIVE model.
# (This is XGBoost's unique pruning parameter — GBM
# doesn't have this)
#
# SUBSAMPLE:
# Percent of data used for each tree (e.g. 0.8 prevents
# overfitting — same idea as Random Forest row sampling)
#
# COLSAMPLE_BYTREE:
# Percent of FEATURES used for each tree (like Random
# Forest's max_features — adds diversity between trees)
#
# QUICK REFERENCE:
# ┌──────────────────┬─────────────────┬──────────────────┐
# │ Hyperparameter   │ Increase →      │ Decrease →       │
# ├──────────────────┼─────────────────┼──────────────────┤
# │ learning_rate    │ Faster but      │ Slower but       │
# │                  │ overfit risk    │ more stable      │
# │ gamma            │ Conservative,   │ More splits,     │
# │                  │ underfitting    │ overfitting risk │
# │ subsample        │ More data/tree  │ More diversity   │
# │ colsample_bytree │ More features   │ More diversity   │
# │ max_depth        │ Overfitting     │ Underfitting     │
# └──────────────────┴─────────────────┴──────────────────┘
# ============================================================


# ============================================================
# GBM vs XGBOOST — FULL COMPARISON
#
# ┌──────────────────┬──────────────────┬───────────────────┐
# │ Feature          │ Gradient Boosting│ XGBoost           │
# ├──────────────────┼──────────────────┼───────────────────┤
# │ Speed            │ Standard         │ Fast (Parallelized│
# │                  │                  │ all CPU cores)    │
# │ Overfitting      │ Easy to overfit  │ Harder — built-in │
# │                  │                  │ L1/L2 regularize  │
# │ Missing Data     │ Must handle      │ Handles           │
# │                  │ manually         │ automatically     │
# │ Pruning          │ Top-down         │ Bottom-up (more   │
# │                  │ (stops early)    │ thorough)         │
# │ Unique param     │ None             │ gamma (gain-based │
# │                  │                  │ pruning)          │
# └──────────────────┴──────────────────┴───────────────────┘
# ============================================================


# ============================================================
# CODE
# ============================================================

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# 1. Split data — NO scaling needed!
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 2. Initialize XGBoost
# n_estimators  → number of trees (use early stopping!)
# max_depth     → keep low (3 to 6)
# learning_rate → step size (0.01 to 0.3)
# n_jobs=-1     → use ALL CPU cores
xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    n_jobs=-1
)

# 3. Fit with Early Stopping
# Stops the model if it stops improving on test set
# This prevents overfitting WITHOUT manually tuning
# n_estimators — it stops at the optimal number!
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# 4. Predict
predictions = xgb_model.predict(X_test)
print("XGBoost Test Score:", xgb_model.score(X_test, y_test))


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. XGBoost = Gradient Boosting + Regularization +
#    Parallelism + Auto Missing Value Handling
#
# 2. DO NOT SCALE DATA before XGBoost — it has internal
#    binning that works best on raw distributions. Scaling
#    can actually HURT performance.
#
# 3. Similarity Score + Gain → XGBoost's way of deciding
#    splits (replaces Gini/Entropy)
#    Low Gain → branch PRUNED (removed entirely)
#
# 4. gamma is unique to XGBoost — minimum gain required
#    to make a split. Higher gamma = fewer splits =
#    simpler model
#
# 5. Early stopping → set n_estimators=1000 and let
#    XGBoost stop automatically when test score stops
#    improving — finds the optimal n_estimators for free
#
# 6. INTERVIEW TRAP:
#    Q: "What is the difference between XGBoost and
#        standard Gradient Boosting?"
#    Right answer: "Three key differences. First, XGBoost
#    has L1 and L2 regularization built into the objective
#    function, making it much harder to overfit. Second,
#    it's parallelized across all CPU cores so it's
#    significantly faster. Third, it handles missing values
#    automatically by learning the optimal default direction
#    at each split — standard GBM requires manual
#    imputation. XGBoost also uses a Gain-based pruning
#    system with the gamma parameter which GBM doesn't have."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Model Deployment
#    Saving your trained model with joblib/pickle and
#    deploying it on Streamlit for your Resume Screener
#    and Plant Disease Detector projects.
# ============================================================