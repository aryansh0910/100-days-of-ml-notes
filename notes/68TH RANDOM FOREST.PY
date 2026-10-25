# ============================================================
# RANDOM FOREST
# ============================================================

# ============================================================
# WHAT IS IT?
#
# A Random Forest is essentially "BAGGING ON STEROIDS."
# While Bagging creates a committee of trees, Random Forest
# ensures that the committee is composed of DIVERSE THINKERS
# by adding a layer of randomness to the FEATURES themselves.
#
# LIKE IN BAGGING U SELECT RANDOM ROWS FOR EACH TREE, IN
# RANDOM FOREST U ALSO SELECT RANDOM COLUMNS AS WELL AS
# ROWS FOR EACH TREE, SO EACH TREE SEES A DIFFERENT
# PERSPECTIVE OF THE DATA, AND THEN WE AVERAGE THEIR
# PREDICTIONS. This makes Random Forests more powerful and
# less prone to overfitting than a single Decision Tree.
# ============================================================


# ============================================================
# 1. THE "RANDOM" IN RANDOM FOREST
#
# In a standard Bagging ensemble, every tree is allowed to
# look at EVERY FEATURE (BMI, Age, BP, etc.) to make a
# split. This often leads to "COPYCAT TREES" where every
# tree starts by splitting on the most obvious feature
# (like BMI).
#
# THE RANDOM FOREST TWIST:
# When a tree needs to make a SPLIT, it is only allowed to
# see a RANDOM SUBSET of the features.
#
# Tree 1 might only be allowed to choose between Age and BP
# Tree 2 might only see BMI and S1
#
# This forces the forest to explore "HIDDEN" patterns that
# a single greedy tree would ignore.
#
# VISUAL — Copycat Trees vs Random Forest:
#
# BAGGING (copycat problem):
# Tree 1: first split → BMI (strongest feature, always picked)
# Tree 2: first split → BMI (same thing again)
# Tree 3: first split → BMI (all trees look the same!)
# → 100 nearly identical trees → ensemble doesn't help much
#
# RANDOM FOREST (forced diversity):
# Tree 1: can only see [Age, BP] → first split → Age
# Tree 2: can only see [BMI, S1] → first split → BMI
# Tree 3: can only see [BP, S2]  → first split → BP
# → 100 DIFFERENT trees → errors cancel out ✅
# ============================================================


# ============================================================
# THE KEY DIFFERENCE — BAGGING vs RANDOM FOREST
#
# THE SIMPLEST WAY TO REMEMBER IT:
# RANDOM FOREST = BAGGING + FEATURE RANDOMIZATION
#
# BAGGING JUST RANDOMLY SELECTS THE ROWS FOR EACH TREE.
#
# RANDOM FOREST ALSO RANDOMLY SELECTS THE COLUMNS (FEATURES)
# FOR EACH TREE, SO EACH TREE SEES A DIFFERENT PERSPECTIVE
# OF THE DATA.
#
# THE MOST CRITICAL DISTINCTION:
# COLUMN SAMPLING IN BAGGING → TREE LEVEL
# (same fixed set of columns for ALL splits in that tree)
#
# COLUMN SAMPLING IN RANDOM FOREST → SPLIT LEVEL
# (different random columns considered at EACH INDIVIDUAL
#  SPLIT within the same tree)
#
# SO EVEN IF U USE COLUMN SAMPLING IN BAGGING, U CAN SEND
# ONLY THE SELECTED COLUMNS IN ALL THE TREES AND THAT WOULD
# BE THE SAME SET. BUT IN RANDOM FOREST, EACH SPLIT GETS A
# FRESH RANDOM SAMPLE OF COLUMNS — MUCH MORE RANDOMNESS!
#
# ALSO: BAGGING CAN HAVE OTHER MODELS AS WELL LIKE KNN,
# SVM, LOGISTIC REGRESSION ETC. BUT RANDOM FOREST IS JUST
# AN ENSEMBLE OF DECISION TREES WITH FEATURE RANDOMIZATION.
#
# VISUAL:
#
# BAGGING (tree-level column sampling):
# Tree 1: columns fixed as [Age, BMI] for ALL its splits
#   Split 1 → chooses from [Age, BMI]
#   Split 2 → chooses from [Age, BMI]  ← same set always
#   Split 3 → chooses from [Age, BMI]
#
# RANDOM FOREST (split-level column sampling):
# Tree 1: each split gets fresh random columns
#   Split 1 → chooses from [Age, BP]   ← random pick
#   Split 2 → chooses from [BMI, S1]   ← different random pick
#   Split 3 → chooses from [Age, S2]   ← different again
# ← much more diversity within the same tree!
# ============================================================


# ============================================================
# 3. WHY RANDOM FORESTS ARE "SMARTER" THAN SINGLE TREES
#
# LOWER VARIANCE: Because it averages 100+ trees, it
# doesn't get "distracted" by outliers or noise in the data.
#
# FEATURE INTERACTION: Excellent at finding how features
# work TOGETHER (e.g. how Age affects the impact of BMI).
#
# HANDLES NON-LINEARITY: Can fit any shape of data by
# creating enough "steps" with its trees — unlike Linear
# Regression which assumes a straight line.
#
# HANDLES MISSING DATA: Generally very robust even if
# your dataset isn't perfect.
# ============================================================


# ============================================================
# HOW RANDOM FOREST WORKS SO WELL
#
# THE SECRET: "The Wisdom of the Diverse Crowd"
#
# 1. THE POWER OF "INDEPENDENCE":
# In statistics, if you average many INDEPENDENT errors,
# they eventually cancel each other out.
#
# A Single Tree is prone to High Variance — it "memorizes"
# the noise in the training data (overfitting). If the data
# has a weird outlier, the tree will grow a branch just for
# that outlier.
#
# The Forest creates 100+ trees that are PURPOSEFULLY
# DIFFERENT. Because each tree is "wrong" in a DIFFERENT WAY,
# when you average them, the errors disappear, leaving only
# the TRUE SIGNAL.
#
# 2. THE BIAS-VARIANCE TRADEOFF:
# Random Forest is a master at reducing VARIANCE without
# significantly increasing BIAS.
#
l# VARIANCE: How much the model changes if you change the
# training data. (Single trees have HIGH variance)
#
# BIAS: How much the model's assumptions differ from
# real-world truth.
#
# By averaging the trees, Random Forest "smooths out" the
# jagged edges of individual decision trees, creating a
# much more stable "prediction surface."
#
# IT SMOOTHENS THE BOUNDARIES OF THE DECISION TREES,
# MAKING IT MORE LIKELY TO GENERALIZE WELL TO NEW DATA.
# ============================================================


# ============================================================
# 4. HYPERPARAMETERS TO WATCH
#
# n_estimators: How many trees? (Usually 100 to 500.
#               More is usually better, but slower.)
#
# max_features: How many features should each tree see?
#               (THE "RANDOM" PART — FEATURES ARE THE
#               COLUMNS IN THE DATASET, SO THIS CONTROLS
#               HOW MANY COLUMNS EACH SPLIT GETS TO SEE.)
#
# n_jobs: Set to -1 to use ALL your CPU cores. This makes
#         training much faster on your PC!
#
# oob_score: Random Forest has a built-in "Self-Test."
#            It tests itself on the data it DIDN'T use
#            during training (the 37% OOB rows).
# ============================================================


# ============================================================
# CODE
# ============================================================

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# ── CLASSIFICATION ──────────────────────────────────────────
X, y = make_classification(n_samples=10000, n_features=10,
                            n_informative=3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_clf = RandomForestClassifier(
    n_estimators=500,   # 500 trees in the forest
    max_features='sqrt',# each split sees √n_features columns
                        # (most common default for classification)
    oob_score=True,     # free self-test on unseen 37% data
    n_jobs=-1,          # use all CPU cores for speed
    random_state=42
)

rf_clf.fit(X_train, y_train)

print("OOB Score:", rf_clf.oob_score_)
print("Test Accuracy:", accuracy_score(y_test,
                                       rf_clf.predict(X_test)))

# Feature importances — which columns mattered most?
print("Feature Importances:", rf_clf.feature_importances_)
# Random Forest gives SMOOTHER importances than single tree
# because it averages across all trees — more reliable signal

# ── REGRESSION ──────────────────────────────────────────────
diabetes = load_diabetes()
X_d, y_d = diabetes.data, diabetes.target

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42
)

rf_reg = RandomForestRegressor(
    n_estimators=500,
    max_features='sqrt',
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

rf_reg.fit(X_train_d, y_train_d)

print("Regression R2:", r2_score(y_test_d,
                                  rf_reg.predict(X_test_d)))


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Random Forest = Bagging + Feature Randomization at
#    SPLIT LEVEL (not tree level like plain Bagging)
#
# 2. THE KEY DIFFERENCE from Bagging:
#    Bagging → same column set for all splits in a tree
#    Random Forest → fresh random columns at EACH SPLIT
#    → much more diversity → errors cancel out better
#
# 3. Random Forest ONLY uses Decision Trees as base models
#    Bagging can use ANY model (KNN, SVM, etc.)
#
# 4. Feature importances are MORE RELIABLE in Random Forest
#    than in a single tree because they're averaged across
#    hundreds of trees
#
# 5. n_jobs=-1 → uses all CPU cores → much faster training
#
# 6. No scaling needed — same reason as single Decision Tree
#    (threshold-based splits, not distance-based)
#
# 7. INTERVIEW TRAP:
#    Q: "What is the difference between Bagging and
#        Random Forest?"
#    Right answer: "Both use the same model on different
#    bootstrap samples of rows. The key difference is
#    WHERE feature randomness is applied. In Bagging, you
#    can randomly sample columns but it's fixed per tree —
#    all splits in that tree use the same column set. In
#    Random Forest, a fresh random subset of columns is
#    considered at EVERY INDIVIDUAL SPLIT, creating much
#    more diversity within each tree itself. Also, Random
#    Forest is exclusively Decision Trees — Bagging works
#    with any base model."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Boosting — AdaBoost
#    Unlike Bagging (parallel), Boosting trains trees
#    sequentially, each one fixing the mistakes of the last.
# ============================================================