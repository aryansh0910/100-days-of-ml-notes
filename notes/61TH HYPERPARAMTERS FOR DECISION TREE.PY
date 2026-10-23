# ============================================================
# DECISION TREE HYPERPARAMETERS
# ============================================================

# ============================================================
# WHAT IS IT?
#
# Hyperparameters are the "RULES OF GROWTH" for a Decision
# Tree. Without them, a tree will grow until it memorizes
# every single row of your data, leading to Overfitting.
#
# By setting these, you are essentially "PRUNING" the tree
# — telling it when to STOP asking questions.
# ============================================================


# ============================================================
# 1. max_depth — THE "HEIGHT" OF THE TREE
#
# MEANS AFTER ROOT NODE HOW MANY NODES WILL BE MADE FURTHER
# — LIKE HOW MANY CUTS WILL BE MADE.
#
# This is the most common hyperparameter. It limits how
# many LAYERS of questions the tree can have.
#
# LOW DEPTH (e.g. 2 or 3): The tree is simple, easy to read,
# but might UNDERFIT (miss details).
#
# HIGH DEPTH (e.g. 20 or None): The tree captures every
# tiny detail, including noise, leading to OVERFITTING.
#
# VISUAL — depth comparison:
#
# max_depth=2 (simple, may underfit):
#         [Root]
#        /      \
#    [Node]    [Node]
#    /   \      /   \
#  Leaf Leaf  Leaf  Leaf      ← only 4 final groups
#
# max_depth=10 (complex, may overfit):
#         [Root]
#        /      \
#    [Node]    [Node]
#     ...        ...           ← many layers deep
#      \          /
#    hundreds of tiny leaves   ← each leaf has very
#    (memorizing individual      few/single data points
#     data points)
# ============================================================


# ============================================================
# 2. min_samples_split
#
# HIGHER VALUE WILL GIVE UNDERFITTING AND LOWER VALUE WILL
# GIVE OVERFITTING.
#
# This sets the MINIMUM number of data points (samples)
# that must be in a node before the tree is ALLOWED to
# split it further.
#
# EXAMPLE: If min_samples_split = 10, and a box only has
# 8 people in it, the tree is FORBIDDEN from drawing any
# more lines inside that box.
#
# Higher values prevent the tree from creating rules based
# on very small, statistically insignificant groups.
#
# VISUAL:
#
# Node with 8 samples, min_samples_split=10:
# ┌─────────────────┐
# │ 8 samples        │  ← can't split further, becomes
# │ (below threshold)│     a leaf as-is, even if mixed
# └─────────────────┘
#
# Node with 50 samples, min_samples_split=10:
# ┌─────────────────┐
# │ 50 samples       │  ← allowed to split further,
# │ (above threshold)│     tree keeps asking questions
# └─────────────────┘
# ============================================================


# ============================================================
# 3. min_samples_leaf
#
# ITS SIMILAR TO min_samples_split.
#
# This sets the minimum number of samples that a LEAF NODE
# (the final box) must have.
#
# INTUITION: It smooths the model. If a split results in a
# leaf with only 1 person, that split is likely just
# capturing NOISE. Setting this to 5 or 10 ensures every
# "prediction box" is based on a decent group of people.
#
# DIFFERENCE FROM min_samples_split:
# min_samples_split → checks BEFORE allowing a split
# (does the parent node have enough samples to split?)
#
# min_samples_leaf → checks AFTER the split happens
# (will the resulting leaves each have enough samples?)
#
# VISUAL:
#
# Attempted split with min_samples_leaf=5:
#
#       [Node: 20 samples]
#       /                \
#  [Leaf: 17 samples]  [Leaf: 3 samples]  ← REJECTED!
#                                            only 3 samples,
#                                            below minimum
#                                            of 5 — this
#                                            split is BLOCKED
# ============================================================


# ============================================================
# 5. max_features
#
# This limits the number of FEATURES the tree looks at
# when searching for the best split.
#
# If you have 20 columns, setting max_features='sqrt'
# means the tree only considers 4 random columns (√20 ≈ 4.5
# rounded) at EACH split.
#
# This is very useful for RANDOM FORESTS to ensure the
# trees don't all look the same.
#
# VISUAL — why this matters for Random Forest:
#
# Without max_features (all trees see all 20 columns):
# Tree 1: always picks "Income" first (strongest feature)
# Tree 2: always picks "Income" first (strongest feature)
# Tree 3: always picks "Income" first (strongest feature)
# → All trees look almost IDENTICAL, ensemble doesn't help much
#
# With max_features='sqrt' (each split sees only 4 random
# columns):
# Tree 1: "Income" wasn't in its random 4 → picks "Tenure"
# Tree 2: "Income" WAS in its random 4 → picks "Income"
# Tree 3: picks "Monthly_Bill" instead
# → Trees become DIVERSE, ensemble of diverse trees is
#   much stronger than ensemble of identical trees
# ============================================================


# ============================================================
# CODE — GRIDSEARCHCV FOR DECISION TREE
# ============================================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Define the "Growth Rules" to test
param_grid = {
    'max_depth'        : [3, 5, 10, None],
    'min_samples_split': [2, 5, 10, None],
    'min_samples_leaf' : [1, 2, 5, None],
    'criterion'        : ['gini', 'entropy']
}

# Setup the search
grid_search = GridSearchCV(DecisionTreeClassifier(),
                           param_grid, cv=5)

# Fit it to find the best rules
grid_search.fit(X_train, y_train)

print("Best Rules found:", grid_search.best_params_)
# GridSearchCV tries every combination of these growth
# rules and tells you which combo generalizes best


# ============================================================
# QUICK REFERENCE — UNDERFIT vs OVERFIT DIRECTION
#
# ┌───────────────────┬─────────────────┬──────────────────┐
# │ Hyperparameter    │ Increase value  │ Decrease value   │
# ├───────────────────┼─────────────────┼──────────────────┤
# │ max_depth         │ → Overfitting   │ → Underfitting   │
# │ min_samples_split │ → Underfitting  │ → Overfitting    │
# │ min_samples_leaf  │ → Underfitting  │ → Overfitting    │
# │ max_features      │ → Overfitting   │ → Underfitting   │
# │                   │   (more choice) │   (more diverse, │
# │                   │                 │    less greedy)  │
# └───────────────────┴─────────────────┴──────────────────┘
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. max_depth limits how many LAYERS of splits the tree
#    can have. High depth = risk of overfitting.
#
# 2. min_samples_split = minimum samples needed BEFORE a
#    node is allowed to split. Higher = more conservative
#    (underfitting risk).
#
# 3. min_samples_leaf = minimum samples a LEAF must end up
#    with AFTER a split. Higher = smoother, less noisy
#    predictions.
#
# 4. max_features limits how many columns the tree considers
#    at EACH split — critical for making Random Forest trees
#    diverse from each other.
#
# 5. All of these are essentially different ways to "prune"
#    the tree — preventing it from memorizing the training data.
#
# 6. INTERVIEW TRAP:
#    Q: "What's the difference between min_samples_split
#        and min_samples_leaf?"
#    Right answer: "min_samples_split checks the PARENT
#    node before allowing a split to happen at all. 
#    min_samples_leaf checks the CHILD nodes AFTER a split
#    — even if the parent had enough samples to split,
#    the split is rejected if it would create a leaf with
#    too few samples. min_samples_leaf gives finer control
#    over the final prediction groups."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Random Forest
#    Combining many Decision Trees into an ensemble to
#    reduce overfitting and improve accuracy.
# ============================================================