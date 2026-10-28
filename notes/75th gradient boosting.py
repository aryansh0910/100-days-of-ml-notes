# ============================================================
# GRADIENT BOOSTING (GBM)
# ============================================================

# ============================================================
# WHAT IS IT?
#
# THE LOSS FUNCTION IN GRADIENT BOOSTING IS THE ACTUAL
# VALUE - THE PREDICTED VALUE, AND THIS IS CALLED THE
# RESIDUAL.
# IN GRADIENT BOOSTING WE TRAIN THE NEXT TREE TO PREDICT
# THE RESIDUAL OF THE PREVIOUS TREE, AND THEN WE ADD THIS
# NEW PREDICTION TO THE PREVIOUS PREDICTION TO GET A
# BETTER PREDICTION.
# THE RESIDUAL STARTS TO BECOME SMALLER AND SMALLER AS WE
# ADD MORE TREES, AND THE FINAL MODEL IS THE SUM OF ALL
# THE TREES' PREDICTIONS.
#
# If AdaBoost is the "Grandfather" of boosting, Gradient
# Boosting (GBM) is the MODERN POWERHOUSE. It is the
# foundation for the most famous algorithms in data science
# like XGBoost, LightGBM, and CatBoost.
#
# IN ADABOOST U USE DECISION STUMPS (DECISION TREE WITH
# ONE MAX DEPTH) AS WEAK LEARNERS, BUT IN GRADIENT
# BOOSTING U CAN USE ANY TYPE OF MODEL AS A WEAK LEARNER,
# BUT THE MOST COMMONLY USED ONE IS DECISION TREES.
#
# While AdaBoost tries to fix mistakes by giving "hard"
# data points MORE WEIGHT, Gradient Boosting fixes
# mistakes by calculating the RESIDUALS (the distance
# between the actual truth and the current prediction).
# ============================================================


# ============================================================
# 1. THE "RESIDUAL" LOGIC
#
# Think of Gradient Boosting as a SCULPTOR working on a
# statue:
#
# MODEL 1 (The Block): Makes a very simple rough guess
# (like the average of all house prices).
#
# THE ERROR (Residual): Calculate how far off that guess
# was for every single point.
# Residual = Actual Value - Predicted Value
#
# MODEL 2 (The Chisel): Instead of predicting the house
# price, this tree is trained to predict the ERROR of
# Model 1.
#
# THE UPDATE: We ADD Model 2's prediction to Model 1's
# prediction. Now the "statue" looks a bit more like a
# person.
#
# REPEAT: Model 3 predicts the remaining error of Model 2,
# and so on.
#
# VISUAL — residuals shrinking over iterations:
#
# Actual house price = 300k
#
# Model 1 prediction: 250k
# Residual 1: 300 - 250 = 50k  ← Model 2 must predict this
#
# Model 2 prediction of residual: 35k
# Combined: 250 + 35 = 285k
# Residual 2: 300 - 285 = 15k  ← Model 3 must predict this
#
# Model 3 prediction of residual: 10k
# Combined: 285 + 10 = 295k
# Residual 3: 300 - 295 = 5k   ← getting smaller! ✅
#
# ...after 100 trees → prediction very close to 300k ✅
#
# Each tree predicts a SMALLER and SMALLER gap until
# the residual is near zero.
# ============================================================


# ============================================================
# 3. WHY IS IT CALLED "GRADIENT"?
#
# In calculus, the GRADIENT tells you which direction to
# move to reach the BOTTOM OF A HILL (the minimum error).
#
# In Gradient Boosting, each new tree is essentially
# moving the model ONE STEP DOWN the "Loss Function hill."
#
# Model 1 → gets us ON the hill (rough prediction)
# Model 2 → looks at the gradient (slope of the error)
#            and moves us DOWNWARD
# Final Model → sits at the very BOTTOM where error is
#               lowest
#
# VISUAL — descending the loss hill:
#
# Loss
#  |●  ← Model 1 (high loss, rough guess)
#  | ●  ← Model 2 (corrected residual)
#  |  ●  ← Model 3
#  |   ●●●●  ← converging to minimum
#  |___________________________ n_estimators
#
# This is literally Gradient Descent applied to an
# ensemble of trees — hence "Gradient Boosting"
# ============================================================


# ============================================================
# ADABOOST vs GRADIENT BOOSTING — COMPARISON
#
# ┌──────────────────┬──────────────────┬───────────────────┐
# │ Feature          │ AdaBoost         │ Gradient Boosting │
# ├──────────────────┼──────────────────┼───────────────────┤
# │ How it fixes     │ Increases weights│ Predicts the      │
# │ errors           │ of "hard" points │ Residual (the gap)│
# │ Flexibility      │ Limited to       │ Can use ANY       │
# │                  │ specific loss    │ differentiable    │
# │                  │ functions        │ loss (MAE, MSE..) │
# │ Sensitivity      │ Very sensitive   │ More robust —     │
# │                  │ to outliers      │ gradients "smooth"│
# │                  │                  │ the path          │
# │ Base learner     │ Decision Stumps  │ Shallow trees     │
# │                  │ (max_depth=1)    │ (max_depth=3-5)   │
# └──────────────────┴──────────────────┴───────────────────┘
# ============================================================


# ============================================================
# 5. HYPERPARAMETERS TO WATCH
#
# LEARNING_RATE (Shrinkage):
# This is VITAL. If you set it to 0.1, each tree only
# contributes 10% to the final answer. This forces the
# model to learn SLOWLY and prevents overfitting.
# (Same inverse relationship with n_estimators as AdaBoost)
#
# SUBSAMPLE:
# If you set this to 0.8, each tree only uses 80% of the
# data. This is called STOCHASTIC GRADIENT BOOSTING, and
# it makes the model much more robust to noise (similar
# to Random Forest's row sampling).
#
# LOSS:
# You can choose 'ls' (Least Squares) or 'lad' (Least
# Absolute Deviation) if you have outliers you want to
# ignore.
#
# MAX_DEPTH:
# GBM usually uses SHALLOW trees (depth 3 to 5) — unlike
# Random Forest which grows deep trees. Shallow trees are
# weak enough to benefit from boosting.
# ============================================================


# ============================================================
# CODE
# ============================================================

from sklearn.ensemble import GradientBoostingRegressor

# 1. Initialize Gradient Boosting
# n_estimators  → Number of "chisel strokes" (trees)
# learning_rate → How much we trust each new tree
#                 (usually small, like 0.1)
# max_depth     → GBM usually uses shallow trees (3 to 5)
gbm = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# 2. Train
gbm.fit(X_reg, y_reg)

# 3. Predict
gbm_pred = gbm.predict(X_reg[:1])
print(f"GBM Prediction: {gbm_pred[0]}")


# ============================================================
# SUMMARY
#
# Gradient Boosting is often MORE ACCURATE than Random
# Forest, but it is HARDER TO TUNE. If you get the
# learning rate or number of trees wrong, it will overfit
# your data much faster than a Random Forest would.
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Core idea: each new tree predicts the RESIDUAL
#    (actual - predicted) of the previous model, not the
#    original target
#
# 2. Residuals get SMALLER with each tree — model
#    converges to the true answer gradually
#
# 3. Called "Gradient" because each tree follows the
#    gradient (slope) of the loss function downhill
#
# 4. AdaBoost vs GBM:
#    AdaBoost → reweights data points
#    GBM      → fits on residuals directly
#    GBM is more flexible and robust to outliers
#
# 5. subsample < 1.0 → Stochastic GBM → more robust,
#    similar to Random Forest's row sampling
#
# 6. Use shallow trees (max_depth=3-5) — deep trees
#    make each step too aggressive, causes overfitting
#
# 7. INTERVIEW TRAP:
#    Q: "What is the difference between AdaBoost and
#        Gradient Boosting?"
#    Right answer: "Both are sequential boosting methods
#    but they fix errors differently. AdaBoost increases
#    the WEIGHT of misclassified points so the next model
#    focuses on them. Gradient Boosting calculates the
#    RESIDUAL (actual - predicted) and trains the next
#    tree to predict that residual directly. GBM is more
#    flexible (works with any differentiable loss function)
#    and more robust to outliers than AdaBoost."
# ============================================================

                                                                                                                                                             
# ============================================================
# COMING UP NEXT:
# → XGBoost
#    The optimized, faster, and more powerful version of
#    Gradient Boosting that dominates ML competitions.
# ============================================================