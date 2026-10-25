# ============================================================
# BAGGING — BOOTSTRAP AGGREGATING
# ============================================================

# ============================================================
# WHAT IS IT?
#
# Bagging is short for Bootstrap Aggregating. It is a
# powerful ensemble technique designed to REDUCE VARIANCE
# (overfitting).
#
# U USE SAME MODEL BUT DIFFERENT DATA, SO EACH MODEL SEES
# A DIFFERENT RANDOM SAMPLE OF THE DATA, AND THEN WE
# AVERAGE THEIR PREDICTIONS.
#
# If a single Decision Tree is like an "emotional" person
# who changes their mind based on every tiny detail,
# Bagging is like a "Committee" of 100 people who are each
# shown a slightly different version of the same story.
# Their collective average is much more stable.
#
# WHILE PREDICTING WE WILL GIVE THE DATA TO BE PREDICTED
# TO ALL THE MODELS AND THEN:
# → Average their predictions (FOR REGRESSION)
# → Take a vote (FOR CLASSIFICATION)
#
# U CAN PERFORM BAGGING BY CHOOSING DIFFERENT ROWS OR
# COLUMNS OR BOTH ROWS AND COLUMNS, SO EACH MODEL SEES A
# DIFFERENT PERSPECTIVE OF THE DATA.
# ============================================================


# ============================================================
# 1. THE TWO STEPS OF BAGGING
#
# STEP 1 — BOOTSTRAPPING (SAMPLING):
# The model creates multiple subsets of your data using
# SAMPLING WITH REPLACEMENT.
#
# Imagine a bag of 10 marbles. You pick one, write down
# its color, and put it BACK. You do this 10 times. Your
# new "Bootstrap" bag might have the "Red" marble three
# times and never see the "Blue" one.
#
# This ensures every tree in the ensemble sees a slightly
# different "perspective" of the data.
#
# VISUAL — Bootstrap Sampling:
#
# Original data (10 rows): [1,2,3,4,5,6,7,8,9,10]
#
# Tree 1 sample: [1,3,3,7,2,9,1,5,8,3]  ← 3 appears 3x, 4 missing
# Tree 2 sample: [2,5,6,6,1,8,10,2,7,5] ← 6 appears 2x, 3 missing
# Tree 3 sample: [4,4,9,1,3,7,8,2,4,6]  ← 4 appears 3x, 5 missing
#
# Each tree trained on a DIFFERENT version of the dataset
# → each learns slightly different patterns
#
# STEP 2 — AGGREGATING (COMBINING):
# For Regression   → take the AVERAGE of all predictions
# For Classification → take a MAJORITY VOTE
# ============================================================


# ============================================================
# 3. WHY IS BAGGING BETTER THAN A SINGLE TREE?
#
# REDUCES OVERFITTING:
# If one tree learns a "weird" pattern in the data that is
# just noise, the other 99 trees (who didn't see that
# specific noise) will OUTVOTE it.
#
# NO PRUNING NEEDED:
# Usually, with a single tree, you have to be very careful
# with max_depth. With Bagging, you can let the trees grow
# DEEPER because the averaging process naturally smooths
# out the errors.
#
# VISUAL — how noise gets cancelled:
#
# True pattern: "High income → No Churn"
# Noise in data: 3 rows where high income → Churn (errors)
#
# Single Tree: might learn the noise → WRONG predictions ❌
#
# Bagging (100 trees):
# ~3 trees see those noisy rows → vote "Churn"
# ~97 trees don't see them → vote "No Churn"
# Final: 97 vs 3 = "No Churn" ✅ noise cancelled out!
# ============================================================


# ============================================================
# 5. OUT-OF-BAG (OOB) SCORE
#
# One "cool" feature of Bagging is that you don't even
# need a separate Test set to know if it's working. Since
# each tree only sees about 63% of the data, the remaining
# 37% (the "Out-of-Bag" data) can be used to TEST the tree
# immediately.
#
# VISUAL — OOB concept:
#
# Full dataset: 1000 rows
#
# Tree 1 trained on: rows [1,3,3,7,...] (63% of data)
# Tree 1 tested on:  rows [2,4,6,8,...] (37% NOT seen) ← OOB
#
# Tree 2 trained on: [2,5,6,6,...] → tested on its own OOB
# Tree 3 trained on: [4,4,9,1,...] → tested on its own OOB
#
# OOB Score = average test accuracy across all trees
#             using their own unseen data
# This is basically FREE cross-validation! ✅
# ============================================================


# ============================================================
# CODE — BAGGING CLASSIFIER
#
# THIS WOULD HELP TO REDUCE THE OVERFITTING OF THE MODEL
# AND INCREASE THE ACCURACY OF THE MODEL AS WELL, BECAUSE
# EACH TREE SEES A DIFFERENT PERSPECTIVE OF THE DATA, AND
# THEN WE AVERAGE THEIR PREDICTIONS.
# U CAN USE KNN AS WELL.
# ============================================================

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=10000, n_features=10,
                            n_informative=3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Single Decision Tree (baseline)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print("Decision Tree accuracy", accuracy_score(y_test, y_pred))

# Bagging
bag = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=500,  # NUMBER OF TREES IN THE FOREST
    max_samples=0.25,  # EACH TREE SEES A RANDOM 25% OF THE DATA
    bootstrap=True,    # SAMPLING WITH REPLACEMENT — DATA CAN
                       # REPEAT IN DIFFERENT TREES
    max_features=0.5,  # EACH TREE SEES A RANDOM 50% OF THE
                       # FEATURES (COLUMNS)
    oob_score=True,    # THIS IS LIKE TESTING THE DATA AS THE
                       # MODEL HASN'T YET SEEN IT — GIVES AN
                       # IDEA OF PERFORMANCE WITHOUT NEEDING A
                       # SEPARATE TEST SET
    random_state=42
)

bag.fit(X_train, y_train)

print("Bagging OOB Score:", bag.oob_score_)
# OOB score ≈ test accuracy — if it's similar to your
# test accuracy, the model is NOT overfitting ✅

y_pred_bag = bag.predict(X_test)
print("Bagging Test accuracy",
      accuracy_score(y_test, y_pred_bag))
# Compare this to single tree — bagging should be higher!


# ============================================================
# CODE — BAGGING REGRESSOR
#
# ITS SAME AS THE BAGGING CLASSIFIER JUST U HAVE TO USE THE
# DECISION TREE REGRESSOR INSTEAD OF THE CLASSIFIER AND
# THEN INSTEAD OF ACCURACY SCORE U CAN USE R2 SCORE OR
# MEAN ABSOLUTE ERROR OR MEAN SQUARED ERROR TO CHECK THE
# PERFORMANCE OF THE MODEL.
# ============================================================

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

# U CAN JUST FILL THE VALUES IN PLACE OF BASE ESTIMATOR
# AND OTHER PARAMETERS TO CHECK THE PERFORMANCE
bag_reg = BaggingRegressor(
    base_estimator=DecisionTreeRegressor(),
    bootstrap=True,
    bootstrap_features=False,
    max_features=1.0,
    max_samples=1.0,
    n_estimators=10,
    n_jobs=None,
    oob_score=False,
    random_state=1,
    verbose=0,
    warm_start=False
)


# ============================================================
# KEY PARAMETERS EXPLAINED
#
# n_estimators     → how many trees to build (more = better
#                    but slower, 100-500 is typical)
# max_samples      → what fraction of ROWS each tree sees
#                    (0.25 = 25% of rows per tree)
# max_features     → what fraction of COLUMNS each tree sees
#                    (0.5 = 50% of features per tree)
# bootstrap=True   → sampling WITH replacement (standard
#                    bagging). False = no replacement
# oob_score=True   → use the 37% unseen rows as free
#                    validation set
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Bagging = same model, different data subsets (rows
#    and/or columns), predictions averaged at the end
#
# 2. Bootstrap = sampling WITH replacement — each tree's
#    training set can have duplicates and missing rows
#    → each tree sees ~63% of data, ~37% is OOB
#
# 3. OOB Score = free validation score using the 37% of
#    data each tree never saw — no need for a separate
#    test set
#
# 4. No pruning needed — let trees grow deep, averaging
#    cancels out individual tree noise automatically
#
# 5. Can bag ANY model — Decision Tree, KNN, SVM, etc.
#    base_estimator controls which model is used
#
# 6. INTERVIEW TRAP:
#    Q: "Why does Bagging reduce overfitting when it uses
#        the same model type?"
#    Right answer: "Each tree is trained on a DIFFERENT
#    bootstrap sample, so each tree overfits to slightly
#    different noise. When you AVERAGE across all trees,
#    the noise cancels out because it's random and
#    inconsistent across trees, while the true signal is
#    consistent and gets reinforced. One tree's noise is
#    another tree's missing data."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Random Forest
#    Bagging applied specifically to Decision Trees, with
#    an extra trick — random feature selection at each
#    split to make trees even more diverse.
# ============================================================