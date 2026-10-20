# ============================================================
# RIDGE FOR MULTICOLLINEARITY + LINEAR vs RIDGE vs LASSO
# ============================================================

# ============================================================
# WHAT IS IT?
#
# REGULARIZED RIDGE REGRESSION MEANS ADDING SOMETHING TO
# THE LOSS FUNCTION TO AVOID OVERFITTING OF THE MODEL.
#
# IT IS VERY USEFUL IN CASE WHERE THERE IS MULTICOLLINEARITY
# I.E. TWO COLUMNS ARE HIGHLY CORRELATED AND U CAN'T DROP
# THEM AS THEY ARE NOT EXACTLY THE SAME THING.
# ============================================================


# ============================================================
# THE CASE FOR RIDGE — THE "SMART" WAY
#
# Imagine two features are highly related but not identical.
# For example, in your churn data: "Monthly_Data_Usage" and
# "Monthly_Bill_Amount."
#
# → They are correlated (higher usage usually means a
#   higher bill)
# → BUT they aren't the same thing. The usage might capture
#   a customer's habits, while the bill captures their budget.
#
# IF YOU DROP ONE: You lose the unique "signal" that the
# dropped column provided.
#
# IF YOU USE RIDGE: You keep BOTH. Ridge realizes they are
# "talking over each other" and dampens their coefficients
# so they can coexist without breaking the model.
#
# VISUAL — what dropping loses vs what Ridge preserves:
#
# Drop one column:
# Monthly_Bill weight = 0.8   ← all signal forced into one
# Monthly_Usage weight = (dropped, gone) ← unique info lost
#
# Ridge (keep both, dampened):
# Monthly_Bill weight  = 0.4  ← shares the "talking"
# Monthly_Usage weight = 0.35 ← keeps its unique signal
# Both contribute, neither dominates ✅
# ============================================================


# ============================================================
# 3. THE "MATHEMATICAL HEADACHE" OF DROPPING
#
# When you have 100+ features (common after One-Hot
# Encoding), manually checking every pair for correlation
# is like playing a never-ending game of Whac-A-Mole.
#
# HIDDEN CORRELATION:
# Sometimes, one column isn't correlated with just ONE
# other column, but with a COMBINATION of three others.
# You can't see this on a simple correlation plot.
#
# RIDGE'S ADVANTAGE:
# Ridge handles this automatically through the math. It
# doesn't care if the correlation is simple or complex —
# the L2 penalty stabilizes the whole system at once.
#
# VISUAL — hidden multivariate correlation:
#
# Simple correlation check (correlation matrix):
# Col_A vs Col_B → 0.3  (looks fine, not flagged)
# Col_A vs Col_C → 0.2  (looks fine, not flagged)
# Col_A vs Col_D → 0.25 (looks fine, not flagged)
#
# But Col_A might actually equal (Col_B + Col_C + Col_D)/3
# → a combination relationship invisible to pairwise checks!
# Ridge doesn't need to "see" this — the math handles it
# automatically across ALL features at once.
# ============================================================


# ============================================================
# WHEN TO USE WHICH — LINEAR vs RIDGE vs LASSO
#
# In the real world, Linear Regression is often used as
# the "Baseline." But in most modern machine learning
# competitions and production environments, we almost
# always prefer a regularized model like Ridge.
# ============================================================


# ============================================================
# 1. LINEAR REGRESSION — "THE NATURALIST"
#
# Use this when your data is clean, simple, and you have
# way more rows than columns.
#
# WHEN TO USE: You have a small number of features (<10)
# and they are NOT correlated with each other.
#
# THE GOAL: Inference. You want to be able to say, "A
# 1-unit change in X causes exactly a Y change in the
# outcome," without any mathematical "shrinking" blurring
# the truth.
#
# THE RISK: It is very fragile. If you have even one
# extreme outlier or two correlated features, the
# coefficients will go crazy.
# ============================================================


# ============================================================
# 2. RIDGE REGRESSION — "THE STABILIZER"
#
# Use this as your DEFAULT starting point for almost any
# real-world regression problem.
#
# WHEN TO USE: You have many features, or your features are
# highly correlated (Multicollinearity).
#
# THE GOAL: Prediction Accuracy. You are willing to accept
# a tiny bit of "bias" (making the coefficients slightly
# less accurate) to get a model that is much more stable
# and generalizes better to new data.
#
# THE "VIBE": It keeps everyone on the team but tells the
# loud people to "pipe down." It shrinks coefficients
# toward zero but never makes them exactly zero.
# ============================================================


# ============================================================
# 3. LASSO REGRESSION — "THE MINIMALIST"
#
# Use this when you have a "messy" dataset with way too
# many columns and you want the model to do the cleaning
# for you.
#
# WHEN TO USE: High-dimensional data (e.g. you have 100
# columns but suspect only 10 actually matter).
#
# THE GOAL: Feature Selection. Lasso is ruthless. It will
# look at a useless feature and set its weight to EXACTLY
# ZERO, effectively deleting it from your model.
#
# THE VIBE: It's like a talent scout — if a feature isn't
# a superstar, it gets cut from the team immediately.
# ============================================================


# ============================================================
# QUICK DECISION GUIDE
#
# ┌──────────────┬─────────────────┬───────────────┬──────────────┐
# │              │ Linear Reg.     │ Ridge (L2)    │ Lasso (L1)   │
# ├──────────────┼─────────────────┼───────────────┼──────────────┤
# │ Best for     │ Few, clean,     │ Many or       │ Many columns,│
# │              │ uncorrelated    │ correlated    │ suspect most │
# │              │ features        │ features      │ are useless  │
# │ Goal         │ Inference       │ Prediction    │ Feature      │
# │              │ (exact truth)   │ accuracy      │ selection    │
# │ Coefficients │ Unaffected      │ Shrunk, never │ Shrunk, CAN  │
# │              │                 │ exactly zero  │ become zero  │
# │ Risk         │ Fragile to      │ Slight bias   │ May drop     │
# │              │ outliers/       │ added         │ useful       │
# │              │ correlation     │               │ features too │
# └──────────────┴─────────────────┴───────────────┴──────────────┘
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Ridge solves multicollinearity WITHOUT dropping columns
#    — it dampens correlated coefficients so they coexist
#
# 2. Dropping correlated columns manually loses unique
#    signal that each column carried — Ridge avoids this loss
#
# 3. Ridge handles HIDDEN multivariate correlations
#    automatically — correlations between combinations of
#    columns that simple pairwise correlation checks can't see
#
# 4. Linear Regression → use for inference, small clean
#    datasets, when you need exact coefficient interpretation
#
# 5. Ridge Regression → DEFAULT choice for real-world
#    problems, especially with many or correlated features
#
# 6. Lasso Regression → use when you suspect many features
#    are useless and want automatic feature selection
#
# 7. INTERVIEW TRAP:
#    Q: "Why not just manually drop one of two correlated
#        columns instead of using Ridge?"
#    Right answer: "Two correlated columns aren't identical
#    — each carries some unique signal. Dropping one loses
#    that signal entirely. Ridge keeps both but dampens
#    their coefficients so they coexist without destabilizing
#    the model. Also, with 100+ features, manually checking
#    every pair (and hidden multivariate correlations) is
#    impractical — Ridge handles this automatically through
#    the L2 penalty math."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Lasso Regression (L1)
#    The "ruthless" regularization that can eliminate
#    useless features entirely by shrinking their weights
#    to exactly zero.
# ============================================================