# ============================================================
# ELASTIC NET REGRESSION
# ============================================================

# ============================================================
# WHAT IS IT?
#
# ITS A COMBINATION OF BOTH LASSO AND RIDGE REGRESSION.
# LIKE WHEN THERE ARE 100 COLUMNS AND U DON'T KNOW WHICH
# U CAN KEEP AND WHICH NOT, U CAN USE ELASTIC NET
# REGRESSION IN SUCH CASES.
# ============================================================


# ============================================================
# THE CORE CONCEPT
#
# Sometimes Lasso is too aggressive (it might delete a
# column you actually need), and Ridge is too soft (it
# keeps all the useless noise). Elastic Net bridges this
# gap by using a "Weighted Average" of both penalties.
#
# It is particularly powerful when you have multiple
# correlated features. While Lasso might randomly pick one
# and kill the others, Elastic Net tends to keep them
# together as a GROUP (like Ridge) while still being able
# to zero out truly useless features (like Lasso).
#
# VISUAL — the "best of both" idea:
#
# Lasso alone (too aggressive on correlated pairs):
# Minutes_Used → 0.6  ← kept
# Data_Used    → 0.0  ← randomly killed, even if useful ❌
#
# Ridge alone (too soft on useless features):
# Favorite_Color → 0.0003  ← should be 0 but isn't ❌
#
# Elastic Net (balanced):
# Minutes_Used   → 0.4  ← kept, dampened with its pair
# Data_Used      → 0.35 ← kept, dampened with its pair  ✅
# Favorite_Color → 0.0  ← correctly killed (useless) ✅
# ============================================================


# ============================================================
# THE TWO KEY PARAMETERS
#
# To make it easier to code, Scikit-Learn uses two
# specific parameters:
#
# ALPHA: The overall strength of the penalty (how much
#        discipline to apply)
#
# L1_RATIO: The balance between Lasso and Ridge
#
# If l1_ratio = 1   → It's 100% Lasso
# If l1_ratio = 0   → It's 100% Ridge
# If l1_ratio = 0.5 → It's a 50/50 mix
#
# VISUAL — the dial:
#
# l1_ratio = 0          0.5           1
#    |───────────────────|───────────────────|
#  Pure Ridge        50/50 Mix          Pure Lasso
#  (no zeros,        (some grouping,    (aggressive zeroing,
#   keeps all)        some zeroing)      can lose features)
# ============================================================


# ============================================================
# 3. WHY USE ELASTIC NET FOR CHURN?
#
# Imagine your churn data has two columns: Minutes_Used
# and Data_Used. They are highly correlated.
#
# LASSO might kill Minutes_Used entirely.
#
# RIDGE will keep both but won't simplify the model.
#
# ELASTIC NET will realize they are a GROUP; it will keep
# both but shrink them effectively, and if there's a third
# column like Favorite_Color_of_Customer, it will set that
# one to zero.
# ============================================================


# ============================================================
# CODE
# ============================================================

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV

# 1. Manual Implementation
en = ElasticNet(alpha=0.1, l1_ratio=0.5)
en.fit(X_train_scaled, y_train)

# 2. The Professional Way (Automated Tuning)
# This will find the best alpha AND the best l1_ratio for you
en_cv = ElasticNetCV(
    l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
    cv=5,
    alphas=[0.1, 0.5, 1, 5, 10]
)
en_cv.fit(X_train_scaled, y_train)

print(f"Best L1 Ratio: {en_cv.l1_ratio_}")
print(f"Best Alpha: {en_cv.alpha_}")

# ElasticNetCV tries EVERY combination of l1_ratio and
# alpha using cross validation, then picks the combo that
# gives the best generalization — no manual guessing needed!


# ============================================================
# RIDGE vs LASSO vs ELASTIC NET — FULL COMPARISON
#
# ┌──────────────────┬─────────────┬─────────────┬──────────────────┐
# │                  │ Ridge       │ Lasso       │ Elastic Net      │
# ├──────────────────┼─────────────┼─────────────┼──────────────────┤
# │ Penalty          │ L2 (square) │ L1 (abs.)   │ Mix of L1 + L2   │
# │ Zeros features?  │ Never       │ Yes         │ Yes, but smarter │
# │ Correlated       │ Keeps both, │ Picks one   │ Keeps group      │
# │ features         │ shares      │ randomly    │ together,        │
# │                  │ weight      │             │ shrinks together │
# │ Control          │ alpha only  │ alpha only  │ alpha + l1_ratio │
# │ Best for         │ Many useful │ Few truly   │ Many features    │
# │                  │ features    │ important   │ WITH correlated  │
# │                  │             │ features    │ groups           │
# └──────────────────┴─────────────┴─────────────┴──────────────────┘
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Elastic Net = weighted combination of L1 (Lasso) and
#    L2 (Ridge) penalties
#
# 2. l1_ratio controls the mix:
#    1 = pure Lasso, 0 = pure Ridge, 0.5 = balanced
#
# 3. Best use case: many features WITH correlated groups —
#    fixes Lasso's "randomly kills one of the pair" problem
#    while still zeroing genuinely useless features
#
# 4. ElasticNetCV automatically searches for the best
#    alpha AND l1_ratio combo using cross validation —
#    no manual tuning needed
#
# 5. Scale your data first — same rule applies as Ridge/Lasso
#
# 6. INTERVIEW TRAP:
#    Q: "If Elastic Net combines both, why not just always
#        use it instead of Ridge or Lasso?"
#    Right answer: "Elastic Net adds an extra hyperparameter
#    (l1_ratio) to tune, increasing complexity and training
#    time, especially with ElasticNetCV searching multiple
#    combinations. If you know your data has minimal
#    correlation, plain Lasso or Ridge is simpler and faster.
#    Elastic Net specifically shines when you have BOTH many
#    useless features AND groups of correlated useful ones —
#    a combination Ridge or Lasso alone can't handle well."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Polynomial Regression
#    Extending Linear Regression to fit curved relationships
#    using polynomialality through feature transformation.
# ============================================================