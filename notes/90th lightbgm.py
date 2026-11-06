# ============================================================
# LIGHTGBM — LIGHT GRADIENT BOOSTING MACHINE
# ============================================================

# ============================================================
# WHAT IS IT?
#
# LightGBM (Light Gradient Boosting Machine) is the natural
# next step after XGBoost. It was developed by Microsoft
# and is currently the "SPEED KING" of boosting algorithms.
#
# If XGBoost is a powerful SUV, LightGBM is a FORMULA 1
# CAR — it's faster, uses less memory, and often gets
# better results on massive datasets.
# ============================================================


# ============================================================
# 1. WHY IS IT "LIGHT"?
#
# LightGBM is "light" because of TWO MAIN TRICKS:
#
# A. GOSS (Gradient-based One-Side Sampling):
# It focuses on the data points with LARGE GRADIENTS
# (the ones it's getting WRONG) and IGNORES the easy ones.
# MEANS LIKE U WILL GET THE HIGHEST PERCENTAGE OF THE
# DATA THAT IS GETTING WRONG AND FOCUS ON THAT.
#
# Why this works: Easy points (small gradient) don't
# contribute much to learning anyway. By skipping them,
# LightGBM trains much faster without losing accuracy.
#
# B. EFB (Exclusive Feature Bundling):
# It bundles together features that RARELY APPEAR AT THE
# SAME TIME (like "is_apple" and "is_orange") to reduce
# the number of features it has to look at.
# LIKE IT GIVES A SINGLE NUMBER TO BOTH IN A SINGLE
# COLUMN — INSTEAD OF 0 AND 1 U GIVE 10 TO MALE AND
# 11 TO FEMALE IN A SINGLE COLUMN ONLY.
#
# VISUAL — GOSS vs standard sampling:
#
# Standard GBM (looks at all rows equally):
# Easy rows: ●●●●●●●●●●●●●●●●●●●●  (waste time on these)
# Hard rows: ××××                    (only 4 wrong ones)
#
# LightGBM GOSS (focuses on hard rows):
# Hard rows: ×××× ← full attention ✅
# Easy rows: ●●   ← small random sample only
# Result: same accuracy, much less computation ✅
# ============================================================


# ============================================================
# 2. LEAF-WISE vs LEVEL-WISE GROWTH
#
# This is the BIGGEST TECHNICAL DIFFERENCE between XGBoost
# and LightGBM:
#
# XGBOOST (Level-wise):
# Grows the tree HORIZONTALLY, layer by layer.
# ALL leaves at depth 3 are grown before ANY leaf at
# depth 4.
#
# LIGHTGBM (Leaf-wise):
# Grows the tree by choosing the SINGLE LEAF that reduces
# loss the MOST, regardless of the level.
# LEAF WISE IN LIGHTGBM MEANS IT WILL FIRST COMPLETE A
# BRANCH AND THEN MOVE TO THE NEXT BRANCH.
#
# This makes it much FASTER but also more prone to
# OVERFITTING on small datasets.
#
# VISUAL:
#
# XGBoost (Level-wise, balanced):
#          [Root]
#         /      \
#      [N1]      [N2]     ← depth 1: both grown first
#     /    \    /    \
#   [L1] [L2] [L3] [L4]  ← depth 2: all grown together
#
# LightGBM (Leaf-wise, unbalanced):
#          [Root]
#         /      \
#      [N1]      [N2]
#     /    \
#   [L1] [L2]             ← L2 chosen because it reduces
#          /    \            loss the most → grows deeper
#       [L3] [L4]           while N2 stays as a leaf!
#
# Result: LightGBM creates DEEPER, MORE ASYMMETRIC trees
# that often capture complex patterns better than XGBoost's
# balanced trees.
# ============================================================


# ============================================================
# 4. CATEGORICAL FEATURES — THE "SUPERPOWER"
#
# One of LightGBM's BEST FEATURES is that it doesn't need
# One-Hot Encoding. It can handle CATEGORICAL DATA
# (strings/labels) DIRECTLY, which saves massive memory.
#
# You just have to tell it which columns are categories:
# lgbm.fit(X_train, y_train,
#           categorical_feature=['city', 'gender', 'job_type'])
#
# VISUAL — memory savings:
#
# Without LightGBM (OHE required):
# City column with 500 cities → 500 new binary columns ❌
# → huge sparse matrix, slow training
#
# With LightGBM (direct categorical):
# City column stays as ONE column with category codes ✅
# → LightGBM handles the optimal splitting internally
# ============================================================


# ============================================================
# KEY HYPERPARAMETERS
#
# ┌──────────────────┬──────────────────────────────────────┐
# │ Parameter        │ What it does                         │
# ├──────────────────┼──────────────────────────────────────┤
# │ num_leaves       │ MOST IMPORTANT. Controls complexity. │
# │                  │ Keep < 2^max_depth to avoid overfit  │
# │ min_data_in_leaf │ Prevents overfitting. Set higher     │
# │                  │ (e.g. 100) for small data            │
# │ feature_fraction │ Similar to colsample_bytree in       │
# │                  │ XGBoost (randomly picks features)    │
# │ bagging_fraction │ Similar to subsample (randomly       │
# │                  │ picks rows)                          │
# └──────────────────┴──────────────────────────────────────┘
#
# OVERFITTING FIX FOR LIGHTGBM:
# → Reduce num_leaves (most impactful)
# → Increase min_data_in_leaf
# → Reduce feature_fraction and bagging_fraction
# ============================================================


# ============================================================
# LIGHTGBM vs XGBOOST — COMPARISON
#
# ┌──────────────────┬──────────────────┬───────────────────┐
# │ Feature          │ XGBoost          │ LightGBM          │
# ├──────────────────┼──────────────────┼───────────────────┤
# │ Speed            │ Fast             │ 2-10x FASTER      │
# │ Memory           │ Standard         │ Much LESS RAM     │
# │ Tree growth      │ Level-wise       │ Leaf-wise         │
# │                  │ (balanced)       │ (asymmetric)      │
# │ Categorical data │ Needs OHE first  │ Handles directly  │
# │ Small datasets   │ Better           │ Can overfit       │
# │ Large datasets   │ Good             │ Better (designed  │
# │ (10k+ rows)      │                  │ for this)         │
# │ GOSS             │ No               │ Yes (skips easy   │
# │                  │                  │ data points)      │
# └──────────────────┴──────────────────┴───────────────────┘
# ============================================================


# ============================================================
# CODE
# ============================================================

from lightgbm import LGBMClassifier

# 1. Initialize
# boosting_type='gbdt' → Gradient Boosting Decision Tree
# (standard mode, most common)
lgbm = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    num_leaves=31,      # keep < 2^max_depth = 32 ✅
    random_state=42
)

# 2. Fit
lgbm.fit(X_train, y_train)
# Optional: handle categoricals directly
# lgbm.fit(X_train, y_train,
#           categorical_feature=['city', 'gender'])

# 3. Predict probabilities (for ROC Curve and Thresholds)
y_probs = lgbm.predict_proba(X_test)[:, 1]

# 4. Apply a custom threshold (like we just learned)
custom_threshold = 0.35
lgbm_preds = (y_probs >= custom_threshold).astype(int)


# ============================================================
# SUMMARY
#
# SPEED:  LightGBM is usually 2 to 10 times faster than
#         XGBoost.
#
# MEMORY: Uses much less RAM.
#
# DATA SIZE: Designed for LARGE DATA (10,000+ rows).
# If your data is very small, stick to Random Forest or
# XGBoost to avoid overfitting from leaf-wise growth.
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. LightGBM = XGBoost + two tricks:
#    GOSS → focuses on hard rows, skips easy ones
#    EFB  → bundles rarely-co-occurring features together
#
# 2. Leaf-wise growth (LightGBM) vs Level-wise (XGBoost):
#    Leaf-wise → faster, asymmetric trees, better accuracy
#    Level-wise → slower, balanced trees, safer on small data
#
# 3. num_leaves is the MOST IMPORTANT param in LightGBM
#    Always keep num_leaves < 2^max_depth to avoid overfit
#
# 4. Handles categorical features DIRECTLY — no OHE needed
#    Just pass categorical_feature=['col1', 'col2']
#
# 5. Use LightGBM for large datasets (10k+ rows)
#    Use XGBoost or RF for small datasets
#
# 6. INTERVIEW TRAP:
#    Q: "What is the difference between XGBoost and
#        LightGBM?"
#    Right answer: "Two key differences. First, tree
#    growth strategy — XGBoost grows level-wise (all
#    nodes at the same depth grown together, balanced
#    tree), while LightGBM grows leaf-wise (picks the
#    single leaf with highest loss reduction, creating
#    asymmetric deep trees that are faster and often
#    more accurate). Second, LightGBM uses GOSS to skip
#    easy training points and EFB to bundle rarely-
#    co-occurring features, making it 2-10x faster and
#    more memory efficient. The tradeoff is LightGBM
#    can overfit more easily on small datasets due to
#    its aggressive leaf-wise growth."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → CatBoost
#    Another gradient boosting library — specifically
#    designed for datasets with many categorical features.
# ============================================================