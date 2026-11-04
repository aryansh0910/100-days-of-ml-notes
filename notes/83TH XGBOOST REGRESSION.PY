# ============================================================
# XGBOOST REGRESSION — ADDITIONAL PARAMETERS
# ============================================================

# ============================================================
# NEW PARAMETERS NOT IN PREVIOUS NOTE
#
# OBJECTIVE:
# For regression → defaults to 'reg:squarederror' (MSE)
# For classification → 'binary:logistic' or 'multi:softmax'
# XGBoost automatically picks the right one but you can
# set it explicitly.
#
# ALPHA (L1 Regularization):
# Same as Lasso — can shrink some weights to exactly zero
# Higher alpha → more features ignored → simpler model
#
# LAMBDA (L2 Regularization):
# Same as Ridge — shrinks all weights but keeps them alive
# Higher lambda → all weights smaller → more conservative
#
# EARLY_STOPPING_ROUNDS:
# "If the test error hasn't improved in N rounds, STOP."
# This is a lifesaver — prevents building useless trees
# without manually guessing the right n_estimators
# ============================================================

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

xgb_reg = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    objective='reg:squarederror',  # MSE loss for regression
    alpha=0.1,     # L1 — can zero out useless features
    reg_lambda=1.0,# L2 — shrinks all weights (default=1)
    n_jobs=-1
)

xgb_reg.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

predictions = xgb_reg.predict(X_test)
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2:",  r2_score(y_test, predictions))

# ============================================================
# KEY ADDITIONS TO REMEMBER
#
# 1. objective='reg:squarederror' → explicitly sets MSE
#    loss. Default for regression but good to know the name
#
# 2. alpha (L1) → like Lasso inside XGBoost
#    reg_lambda (L2) → like Ridge inside XGBoost
#    This is the "built-in regularization" that makes
#    XGBoost harder to overfit than standard GBM
#
# 3. early_stopping_rounds → set n_estimators=1000 and
#    let early stopping find the optimal number for you
#    automatically — no manual tuning of n_estimators
#
# 4. INTERVIEW TRAP:
#    Q: "How does XGBoost prevent overfitting?"
#    Right answer: "Three ways simultaneously — gamma
#    prunes branches with insufficient gain, alpha/lambda
#    apply L1/L2 regularization to the weights directly,
#    and early_stopping_rounds stops training when the
#    validation score stops improving. This triple
#    protection is why XGBoost outperforms standard GBM
#    on noisy real-world data."
# ============================================================