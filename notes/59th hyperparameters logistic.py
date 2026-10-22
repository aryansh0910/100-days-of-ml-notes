# ============================================================
# LOGISTIC REGRESSION HYPERPARAMETERS — PENALTY, C, MULTI_CLASS
# ============================================================

# ============================================================
# 1. PENALTY (THE REGULARIZATION TYPE)
#
# Just like in Ridge and Lasso, you can choose how to
# penalize the model to prevent overfitting.
#
# 'l2' (Default): Like Ridge. Keeps all features but
#                 shrinks their weights. Good for general
#                 stability.
#
# 'l1': Like Lasso. Can set some weights to exactly zero.
#       Best for feature selection.
#
# 'elasticnet': A mix of both L1 and L2.
#
# 'none': No penalty (standard Logistic Regression).
#
# This is the SAME logic from Ridge/Lasso/Elastic Net,
# just applied to Logistic Regression's loss function
# (Log Loss) instead of Linear Regression's loss (MSE).
# ============================================================


# ============================================================
# 2. C — THE INVERSE OF STRENGTH
#
# This is the MOST IMPORTANT parameter to tune.
#
# C is the INVERSE of Alpha.
# FORMULA: C = 1 / Alpha
#
# SMALL C (e.g. 0.01): HIGH regularization. The model is
# "pushed" harder to have small weights. Use this if you
# are OVERFITTING.
#
# LARGE C (e.g. 100): LOW regularization. The model is
# "trusted" more to fit the training data closely.
#
# WHY INVERSE? Because in Ridge/Lasso, higher alpha = more
# penalty. But sklearn's Logistic Regression uses C, where
# higher C = LESS penalty (since C = 1/alpha, big C means
# small alpha).
#
# VISUAL — the inverse relationship:
#
# Alpha (Ridge/Lasso logic):    C (Logistic Regression):
# α = 0.01  → light penalty     C = 100   → light penalty
# α = 100   → heavy penalty     C = 0.01  → heavy penalty
#
#  (these behave the SAME way, just labeled oppositely!)
#
# RULE OF THUMB:
# Overfitting (train acc high, test acc low) → DECREASE C
# Underfitting (train acc itself low)        → INCREASE C
# ============================================================


# ============================================================
# 4. MULTI_CLASS
#
# This tells the model how to handle more than two
# categories:
#
# 'ovr' (One-vs-Rest):
# It treats a 3-class problem as THREE SEPARATE "Yes/No"
# problems.
#
# 'multinomial':
# This uses the Softmax logic we just discussed to solve
# all classes AT ONCE.
#
# VISUAL — how OVR works for 3 classes (Basic/Premium/Elite):
#
# Model 1: Is it Basic?    vs (Premium OR Elite)?
# Model 2: Is it Premium?  vs (Basic OR Elite)?
# Model 3: Is it Elite?    vs (Basic OR Premium)?
#
# Each model gives a probability, and the class with the
# HIGHEST probability across all 3 models wins.
#
# vs Softmax (multinomial) — ONE single model computes all
# 3 probabilities together so they sum to exactly 1.0
# (covered in the previous note).
# ============================================================


# ============================================================
# CODE — GRIDSEARCHCV FOR LOGISTIC REGRESSION
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Define the "Grid" of settings to try
param_grid = {
    'C'      : [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver' : ['liblinear']  # 'liblinear' works with both l1 and l2
}

# Setup the search
grid_search = GridSearchCV(LogisticRegression(), param_grid,
                           cv=5, scoring='accuracy')

# Fit it
grid_search.fit(X_train_scaled, y_train)

# View results
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
# GridSearchCV tries EVERY combination of C and penalty
# across 5 folds, then tells you which setup performed best


# ============================================================
# SUMMARY CHECKLIST
#
# If you have too many features → try penalty='l1'
#
# If your model accuracy is LOW on training data → increase C
# (model needs more freedom to fit the data — was too
#  restricted by regularization)
#
# If your model accuracy is HIGH on training but LOW on
# testing → decrease C (model is overfitting, needs more
# regularization to generalize)
#
# If you have more than 2 classes → set
# multi_class='multinomial' and use a compatible solver
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. penalty parameter: 'l2' (default, Ridge-like), 'l1'
#    (Lasso-like, feature selection), 'elasticnet' (mix)
#
# 2. C = 1/alpha — INVERSE relationship with regularization
#    strength. Small C = heavy regularization, Large C =
#    light regularization (opposite intuition from alpha!)
#
# 3. Diagnosis rule:
#    Underfitting → increase C
#    Overfitting  → decrease C
#
# 4. multi_class: 'ovr' trains N separate binary models,
#    'multinomial' (Softmax) trains one unified model
#
# 5. Not all solvers support all penalties — 'liblinear'
#    supports l1+l2, but check compatibility before tuning
#
# 6. INTERVIEW TRAP:
#    Q: "If C is small, does that mean the model has weak
#        regularization?"
#    Wrong: "Yes, small number = small effect"
#    Right: "No — it's the opposite. C = 1/alpha, so a
#    SMALL C means a LARGE alpha internally, which means
#    STRONG (heavy) regularization. This inverse relationship
#    trips a lot of people up — small C = more regularization,
#    not less."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Decision Trees
#    A completely different approach to classification —
#    splitting data based on feature thresholds instead of
#    fitting a line or curve.
# ============================================================