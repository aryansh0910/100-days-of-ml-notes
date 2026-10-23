# ============================================================
# DECISION TREES
# ============================================================

# ============================================================
# WHAT IS IT?
#
# While Logistic Regression tries to draw a single line
# (or curve) to separate data, a Decision Tree asks a
# series of YES/NO questions to split the data into
# smaller and smaller groups.
#
# DECISION TREES IS LIKE A GROUP OF IF ELSE STATEMENTS
# TRYING TO MAKE A GROUP, AND WHEN A PREDICTION IS TO BE
# MADE, IT IS CHECKED WHICH GROUP IS THAT PRESENT IN.
# ============================================================


# ============================================================
# EXAMPLE — CATEGORICAL DECISION TREE
#
#               [ Is it Raining? ]
#               /                \
#             YES                 NO
#            /                     \
# [ Do I have a Gym pass? ]    [ Is it above 30°C? ]
#        /          \               /          \
#      YES           NO           YES           NO
#       |             |            |             |
#  (RUN AT GYM)   (SLEEP)    (RUN AT NIGHT)  (RUN NOW)
# ============================================================


# ============================================================
# ANOTHER EXAMPLE — NUMERICAL DATA
#
#                 [ Mileage < 50,000 km? ]
#                /                      \
#              YES                       NO
#             /                           \
#     [ Age < 3 years? ]           [ Mileage < 100,000 km? ]
#        /          \                   /               \
#      YES           NO               YES                NO
#       |             |                |                  |
#  (HIGH PRICE)  (MID PRICE)      (MID PRICE)        (LOW PRICE)
# ============================================================


# ============================================================
# GEOMETRIC INTUITION — IT MAKES LIKE CUTTING THE SCATTER
# PLOT WITH LINES
#
# Geometric Map: Should I go to the Park?
#
# TEMPERATURE (Celsius)
#  ^
#  |
#  40  +-------------------------+-------------------------+
#      |                         |                         |
#      |        ZONE A           |        ZONE B1          |
#      |    (Temp > 30)          |    (Temp <= 30)         |
#      |                         |    (Is Weekend? YES)    |
#      |    PREDICTION: NO       |    PREDICTION: YES      |
#      |    (Too Hot!)           |    (Perfect Day!)       |
#      |                         |                         |
#  30  +-------------------------+-------------------------+
#      |                         |                         |
#      |        ZONE A           |        ZONE B2          |
#      |    (Temp > 30)          |    (Temp <= 30)         |
#      |                         |    (Is Weekend? NO)     |
#      |    PREDICTION: NO       |    PREDICTION: NO       |
#      |    (Too Hot!)           |    (I have work...)     |
#      |                         |                         |
#  10  +-------------------------+-------------------------+
#      +-------------------------+-------------------------+------> WEEKEND
#      0 (Weekday)               0.5 (Midpoint)            1 (Weekend)
#
# Notice: each split creates a STRAIGHT, AXIS-ALIGNED cut
# (horizontal or vertical line) — unlike Logistic
# Regression's diagonal decision boundary. This is why
# trees handle non-linear patterns well, but always create
# "blocky" rectangular regions.
# ============================================================


# ============================================================
# 1. THE ANATOMY OF A TREE
#
# A Decision Tree is made of three main parts:
#
# ROOT NODE: The very first question (the most important
#            feature). THE MAIN FEATURE AT THE TOP.
#
# DECISION NODES: Internal branches where further questions
#                 are asked. THE MIDDLE FEATURES WHICH ARE
#                 USED TO SPLIT FURTHER AFTER ROOT NODE.
#
# LEAF NODES: The final "buckets." These contain the final
#             prediction (e.g. "Churn" or "Stay"). THE LAST
#             FEATURE, I.E. THE ANSWER.
# ============================================================


# ============================================================
# 2. HOW IT "DECIDES" WHERE TO SPLIT
#
# The tree wants to create "PURE" groups. If a group is
# 100% Churn or 100% Stay, it is perfectly pure. To measure
# this, it uses two main metrics: Gini Impurity and Entropy.
#
# GINI IMPURITY:
# This measures how often a randomly chosen element from
# the set would be INCORRECTLY labeled.
#
# Gini = 0   → Perfect purity (everyone is in the same class)
# Gini = 0.5 → Maximum mess (50/50 split)
#
# FORMULA:
# ┌─────────────────────────────────────────────┐
# │                                             │
# │   Gini = 1 - Σ (probability_of_class)²       │
# │                                             │
# └─────────────────────────────────────────────┘
#
# VISUAL — Gini for different group compositions:
#
# Group A: [Churn, Churn, Churn, Churn]  (100% one class)
# Gini = 1 - (1.0)² = 0  ← perfectly pure ✅
#
# Group B: [Churn, Stay, Churn, Stay]    (50/50 split)
# Gini = 1 - (0.5)² - (0.5)² = 0.5  ← maximum impurity ❌
#
# Group C: [Churn, Churn, Churn, Stay]   (75/25 split)
# Gini = 1 - (0.75)² - (0.25)² = 0.375  ← moderately pure
# ============================================================

# ============================================================
# ENTROPY — ANOTHER WAY TO MEASURE IMPURITY
#
# ENTROPY IS THE MEASURE OF THE DISORDER.
#
# MORE KNOWLEDGE MEANS LESS ENTROPY, I.E. MORE IS THE
# UNCERTAINTY FOR THE ANSWER, MORE IS THE ENTROPY.
#
# SO IN CASE OF BINARY CLASSIFICATION, IF ALL THE ITEMS ARE
# 0 (OR 1), THEN THE ENTROPY WOULD BE 0, AS WE KNOW THAT
# THE ANSWER WILL ALWAYS BE 0. SO FOR THE MOST ENTROPY,
# THE CLASSIFICATION SHOULD HAVE ALMOST EQUAL PROBABILITY
# OF OCCURRING.
#
# VISUAL — entropy vs certainty:
#
# All same class (Entropy = 0):
# [Churn, Churn, Churn, Churn]
# "I'm 100% sure what the answer is" → zero disorder
#
# 50/50 split (Entropy = 1, maximum):
# [Churn, Stay, Churn, Stay]
# "Could genuinely be either" → maximum disorder
#
# Both Gini and Entropy measure the SAME basic idea
# (impurity), just with slightly different formulas. Gini
# is computationally faster (no log), Entropy is more
# mathematically "information theory" based.
# ============================================================


# ============================================================
# INFORMATION GAIN
#
# Information Gain is a metric used to train Decision
# Trees. Specifically, this metric measures the QUALITY
# of a split.
#
# The information gain is based on the DECREASE in entropy
# after a dataset is split on an attribute. Constructing a
# decision tree is all about finding the attribute that
# returns the HIGHEST information gain.
#
# VISUAL:
#
# Before split (parent node):
# Entropy = 1.0 (50/50 mix of Churn/Stay)
#
# After splitting on "Tenure < 12 months?":
# Branch YES: Entropy = 0.2 (mostly Churn) ← much purer!
# Branch NO:  Entropy = 0.3 (mostly Stay)  ← much purer!
#
# Information Gain = Parent Entropy - Weighted Avg of
#                    Children Entropy
#                  = 1.0 - (weighted avg of 0.2, 0.3)
#                  = a LARGE positive number
#
# A large Information Gain means this was a GOOD split —
# the tree picks the feature/threshold that maximizes this
# at every node.
# ============================================================


# ============================================================
# 3. WHY USE DECISION TREES FOR CHURN?
#
# NON-LINEARITY:
# Unlike Logistic Regression, trees don't care if the
# relationship is a straight line. They can capture complex
# patterns (e.g. "Young people churn if they have low data,
# but old people churn if they have high bills").
#
# NO SCALING REQUIRED:
# Trees don't care if Age is 20 and Income is 50,000. They
# just look at "Is Value > 30?" You can SKIP StandardScaler.
#
# HANDLES OUTLIERS:
# One extreme outlier won't ruin the whole model; it just
# gets its own little leaf.
# ============================================================


# ============================================================
# 4. THE BIGGEST DANGER — OVERFITTING
#
# A Decision Tree is "greedy." If you let it, it will keep
# growing until every single customer has their own private
# leaf. This is called MEMORIZING THE DATA.
#
# Training Accuracy: 100% (knows the training data perfectly)
# Testing Accuracy:  Terrible (doesn't understand new data)
#
# To fix this, we use PRUNING or hyperparameters like
# max_depth to stop the tree from growing too large.
#
# VISUAL — overgrown tree vs pruned tree:
#
# OVERGROWN (memorized every customer):
#         [Root]
#        /  |  |  \
#      ... 1000s of tiny leaves, one per customer ❌
#
# PRUNED (max_depth=4, generalizes):
#         [Root]
#        /      \
#    [Node]    [Node]
#    /    \    /    \
#  Leaf  Leaf Leaf  Leaf  ← a handful of meaningful groups ✅
# ============================================================


# ============================================================
# CODE
# ============================================================

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Sample churn data
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'Tenure_Months' : np.random.randint(1, 60, n),
    'Monthly_Bill'  : np.random.randint(20, 150, n),
})
df['Churn'] = ((df['Tenure_Months'] < 12) &
               (df['Monthly_Bill'] > 80)).astype(int)

X = df[['Tenure_Months', 'Monthly_Bill']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# No scaling needed! Trees don't care about feature ranges
tree = DecisionTreeClassifier(
    criterion='gini',   # or 'entropy'
    max_depth=4,         # prevents overfitting
    random_state=42
)

tree.fit(X_train, y_train)

predictions = tree.predict(X_test)
print("Feature importances:", tree.feature_importances_)
# Tells you which feature the tree relied on most


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Three parts of a tree: Root Node (first/most important
#    question), Decision Nodes (further splits), Leaf Nodes
#    (final predictions)
#
# 2. Tree splits create AXIS-ALIGNED cuts (straight
#    horizontal/vertical lines), unlike Logistic Regression's
#    diagonal boundary
#
# 3. Gini Impurity = 1 - Σ(probability²)
#    0 = pure, 0.5 = maximum impurity (for binary)
#
# 4. Entropy = measure of disorder/uncertainty
#    0 = certain, 1 = maximum uncertainty (50/50 split)
#
# 5. Information Gain = decrease in entropy after a split
#    Tree picks the split with HIGHEST information gain
#    at every node
#
# 6. No scaling needed — trees split on thresholds, not
#    distances
#
# 7. BIGGEST RISK: overfitting (memorizing every row).
#    Fix with max_depth, min_samples_split, or pruning
#
# 8. INTERVIEW TRAP:
#    Q: "Why don't Decision Trees need feature scaling?"
#    Right answer: "Trees make decisions by comparing a
#    single feature to a threshold (Is Age > 30?), not by
#    calculating distances between points like KNN or SVM.
#    Since there's no distance calculation, the scale of
#    different features doesn't affect how splits are made."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Random Forest
#    Combining many Decision Trees into an ensemble to
#    reduce overfitting and improve accuracy.
# ============================================================