# ============================================================
# RANDOM FOREST HYPERPARAMETERS
# ============================================================

# ============================================================
# WHAT IS IT?
#
# In RandomForestRegressor or RandomForestClassifier, the
# hyperparameters generally fall into TWO CATEGORIES:
# those that control the FOREST (the collection) and those
# that control the INDIVIDUAL TREES.
# ============================================================


# ============================================================
# 1. FOREST-LEVEL HYPERPARAMETERS
# These control the ENSEMBLE AS A WHOLE.
#
# N_ESTIMATORS: The number of trees in the forest.
# Rule of Thumb: More is usually better for accuracy, but
# it takes more memory and time. Usually, 100 is a great
# start. After a certain point (e.g. 500), you get
# "DIMINISHING RETURNS" where accuracy stops improving
# but your PC gets slower.
#
# VISUAL — diminishing returns:
#
# Accuracy
#  |        ●●●
#  |      ●●
#  |    ●●
#  |  ●●
#  |●●___________________________
#  0   100  200  300  400  500
#       n_estimators
#
# After ~200-300 trees, adding more trees barely helps
# accuracy but keeps increasing training time. Pick the
# "elbow" of this curve.
#
# N_JOBS:
# This tells the computer how many CPU cores to use.
# TIP: Set this to -1. This tells your computer to use
# ALL available processors to train trees IN PARALLEL,
# making it much faster.
#
# RANDOM_STATE:
# A seed number (like 42) to ensure that your results are
# REPRODUCIBLE. If you don't set this, your accuracy might
# change slightly every time you hit "Run."
#
# OOB_SCORE (True/False):
# OOB SCORE IS LIKE THE ROWS WHICH ARE NOT SELECTED IN ANY
# DECISION TREE SO IT CAN BE USED AS A FREE VALIDATION SET
# TO ESTIMATE THE GENERALIZATION SCORE OF THE MODEL WITHOUT
# NEEDING A SEPARATE TEST SET.
#
# IF YOU SET OOB_SCORE TO TRUE, THE RANDOM FOREST WILL USE
# THE OOB SAMPLES TO CALCULATE THE OOB SCORE, WHICH CAN
# GIVE YOU AN IDEA OF HOW WELL YOUR MODEL MIGHT PERFORM ON
# UNSEEN DATA. THIS IS PARTICULARLY USEFUL WHEN YOU HAVE A
# SMALL DATASET AND WANT TO MAXIMIZE THE USE OF YOUR
# TRAINING DATA.
#
# IT IS PROVEN THAT 37% OF THE DATA IS LEFT OUT IN EACH
# TREE, SO YOU CAN EXPECT TO HAVE A GOOD ESTIMATE OF YOUR
# MODEL'S PERFORMANCE WITHOUT NEEDING TO SET ASIDE A LARGE
# PORTION OF YOUR DATA FOR TESTING.
# ============================================================


# ============================================================
# 2. TREE-LEVEL HYPERPARAMETERS
# These control how DEEP and COMPLEX each individual tree
# becomes.
#
# MAX_FEATURES: This is the "MAGIC" of Random Forest.
# 'sqrt' (DEFAULT) → each split sees √n_features columns
# 'log2'           → each split sees log2(n_features) cols
# specific number  → e.g. 5 means exactly 5 columns per split
# This is what makes it "RANDOM" — fresh random column
# subset at EVERY split.
#
# MAX_DEPTH: How deep each tree can grow.
# TIP: In Random Forests, you can often leave this as None
# (let them grow deep), because the AVERAGING PROCESS
# prevents the overfitting that usually happens with deep
# single trees.
#
# MIN_SAMPLES_SPLIT: The minimum number of data points
# required to split a node.
# HIGHER VALUE → prevents the tree from creating branches
# that only apply to one or two specific patients
# (reduces overfitting)
#
# MIN_SAMPLES_LEAF: The minimum number of samples required
# to be at a leaf node (the end of a branch).
#
# QUICK REFERENCE — overfitting vs underfitting direction:
#
# ┌───────────────────┬─────────────────┬──────────────────┐
# │ Hyperparameter    │ Increase →      │ Decrease →       │
# ├───────────────────┼─────────────────┼──────────────────┤
# │ n_estimators      │ more stable     │ more variance    │
# │ max_depth         │ overfitting     │ underfitting     │
# │ min_samples_split │ underfitting    │ overfitting      │
# │ min_samples_leaf  │ underfitting    │ overfitting      │
# │ max_features      │ less random,    │ more random,     │
# │                   │ copycat trees   │ diverse trees    │
# │ max_samples       │ more data/tree  │ less data/tree   │
# └───────────────────┴─────────────────┴──────────────────┘
# ============================================================


# ============================================================
# CODE — GRIDSEARCHCV FOR RANDOM FOREST
# ============================================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100],
    'max_features': ['sqrt', 'log2'],
    'max_depth'   : [4, 6, 8],
    'max_samples' : [0.5, 0.75, 1.0]
    # max_samples tells how many ROWS to use for each tree
    # 0.5 = each tree sees 50% of rows
    # Smaller = more diversity but less data per tree
    # Larger = more data per tree but less diversity
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid=param_grid,
    cv=5
)
grid_search.fit(X_reg, y_reg)

print(f"Best Parameters: {grid_search.best_params_}")
# This tries 2×2×3×3 = 36 combinations × 5 folds = 180 fits
# GridSearchCV picks the combo with the best CV score


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Two categories:
#    Forest-level → n_estimators, n_jobs, random_state,
#                   oob_score, max_samples
#    Tree-level   → max_features, max_depth,
#                   min_samples_split, min_samples_leaf
#
# 2. n_estimators: more trees = more stable but slower.
#    Diminishing returns after ~300-500. Start with 100.
#
# 3. n_jobs=-1 → always set this, uses all CPU cores,
#    trees train in PARALLEL so much faster
#
# 4. max_features is the "Random" in Random Forest —
#    fresh random column subset at EVERY split.
#    'sqrt' is default and works well for most problems.
#
# 5. max_depth=None is often fine in Random Forest —
#    averaging handles the overfitting that deep single
#    trees suffer from
#
# 6. OOB score = free validation using the 37% of data
#    each tree never saw — no need for separate test set
#
# 7. INTERVIEW TRAP:
#    Q: "Should you always set max_depth in Random Forest
#        like you would in a single Decision Tree?"
#    Right answer: "Not necessarily. In a single Decision
#    Tree, max_depth is critical to prevent overfitting.
#    But in Random Forest, the averaging process across
#    hundreds of trees naturally cancels out individual
#    tree overfitting. You can often leave max_depth=None
#    and rely on min_samples_leaf or min_samples_split
#    instead for fine-tuning, since deep diverse trees
#    average out to a stable prediction."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Boosting — AdaBoost
#    Unlike Bagging (parallel trees), Boosting trains
#    trees SEQUENTIALLY, each one focusing on the mistakes
#    of the previous one.
# ============================================================