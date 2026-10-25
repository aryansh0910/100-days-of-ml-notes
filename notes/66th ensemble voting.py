# ============================================================
# VOTING ENSEMBLES
# ============================================================

# ============================================================
# WHAT IS IT?
#
# Voting Ensembles are the most intuitive way to understand
# Ensemble Learning. It's essentially a DEMOCRACY for
# algorithms.
#
# Instead of relying on one "expert" (like a single
# Decision Tree), you gather a COMMITTEE of different
# models (Decision Trees, Logistic Regression, SVM, etc.)
# and let them VOTE on the final answer.
# ============================================================


# ============================================================
# THE FIRST THING YOU SHOULD KNOW ABOUT VOTING ENSEMBLES
# IS THAT THEY ARE ALL ABOUT DIVERSITY.
#
# If all your models are the same (like 100 Decision Trees
# trained on the same data), then you're not really getting
# the benefits of an ensemble. You want to mix it up with
# different algorithms or different subsets of data.
#
# The more DIVERSE your models, the better your ensemble
# will perform.
#
# MEANS UR ALL MODELS SHOULD BE DIFFERENT I.E NOT USING
# THE SAME MATHS BEHIND IT.
#
# AND THE ACCURACY OF EACH MODEL SHOULD BE GREATER THAN
# 50% (FOR CLASSIFICATION) OR BETTER THAN A BASELINE
# (FOR REGRESSION). OTHERWISE, THEY MIGHT HURT THE
# PERFORMANCE OF THE ENSEMBLE RATHER THAN HELP IT.
#
# VISUAL — why diversity matters:
#
# BAD ENSEMBLE (all same type, all make same mistake):
# Tree1: wrong on row 5  ❌
# Tree2: wrong on row 5  ❌  → ensemble STILL wrong on row 5
# Tree3: wrong on row 5  ❌
#
# GOOD ENSEMBLE (diverse, different mistakes):
# DecisionTree: wrong on row 5 ❌, right on rows 1-4 ✅
# LogisticReg:  right on row 5 ✅, wrong on row 3 ❌
# SVM:          right on row 5 ✅, wrong on row 1 ❌
# → ensemble RIGHT on row 5 (2 vs 1 vote) ✅
# ============================================================


# ============================================================
# 1. HARD VOTING vs SOFT VOTING
#
# SOFT IS GENERALLY BETTER THAN HARD VOTING BECAUSE IT
# GIVES MORE WEIGHT TO THE MODELS THAT ARE MORE CONFIDENT
# IN THEIR PREDICTIONS.
#
# HARD VOTING (Majority Rules):
# Each model predicts a CLASS (e.g. "Sick" or "Healthy").
# The class that gets more than 50% of the votes WINS.
#
# SOFT VOTING (Weighted Probabilities):
# Each model gives a PROBABILITY (e.g. "I am 80% sure this
# patient is sick"). We AVERAGE these percentages. This is
# usually more accurate because it gives more weight to the
# models that are "highly confident."
#
# VISUAL — same 3 models, different voting styles:
#
# Model A: 90% Sick, 10% Healthy  ← very confident
# Model B: 55% Sick, 45% Healthy  ← barely sure
# Model C: 60% Sick, 40% Healthy
#
# HARD VOTING:
# A says Sick, B says Sick, C says Sick → 3/3 = SICK
# (treats all three votes equally, ignores confidence)
#
# SOFT VOTING:
# Average: (90+55+60)/3 = 68.3% Sick → SICK
# But now if Model A said 10% Sick:
# Hard: still 2/3 = Sick (B and C)
# Soft: (10+55+60)/3 = 41.7% → HEALTHY ← smarter! ✅
# ============================================================


# ============================================================
# 2. WHY USE VOTING ENSEMBLES?
#
# The secret to a great Voting Ensemble is DIVERSITY.
#
# If you have three models that all make the SAME MISTAKE,
# the ensemble will fail. But if you combine:
# → Decision Tree (good at non-linear patterns)
# → Linear Regression (good at straight-line trends)
# They often CANCEL OUT each other's weaknesses.
# ============================================================


# ============================================================
# 5. THE "NO FREE LUNCH" RULE
#
# Ensembles are powerful, but they are "BLACK BOXES."
#
# Single Tree: Easy to explain (you can see the plot).
#
# Voting Ensemble: Harder to explain because the final
# answer is a mix of three different mathematical
# philosophies.
# ============================================================


# ============================================================
# CODE — VOTING REGRESSOR (DIFFERENT ALGORITHMS)
# ============================================================

import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

# 1. Define our 'Committee' of models
model1 = LinearRegression()
model2 = DecisionTreeRegressor(max_depth=4)
model3 = SVR()

# 2. Put them into the Voting Ensemble
# LIKE THE COLUMN TRANSFORMER, WE GIVE EACH MODEL A NAME
# (LIKE 'lr' FOR LINEAR REGRESSION) AND THEN THE MODEL
# ITSELF (model1)
# U CAN ALSO CHECK EACH MODEL'S ACCURACY BY USING CROSS
# VALIDATION AND THEN CHECK IF UR VOTING IS REALLY
# PERFORMING BETTER OR NOT
voting_model = VotingRegressor(
    estimators=[
        ('lr',  model1),
        ('dt',  model2),
        ('svm', model3)
    ]
)

# 3. Train the entire committee at once
voting_model.fit(X_reg, y_reg)

# 4. Make a prediction
# All three models work behind the scenes and average their
# results

# X_reg[:1] → slice of data keeping 2D shape that sklearn
# needs — even for one row, must be a TABLE not a list
prediction = voting_model.predict(X_reg[:1])

print(f"Ensemble Prediction: {prediction[0]}")
# prediction[0] → .predict() always returns a LIST
# even for one row, so [0] grabs that single number


# ============================================================
# CALCULATION — HOW VotingRegressor AVERAGES
#
# 3. The Calculation: "Averaging"
# Since this is a Regressor (predicting a continuous
# number), the VotingRegressor doesn't pick a "winner."
# It calculates the MEAN (Average).
#
# Final Prediction = (150 + 160 + 140) / 3 = 150
# ============================================================


# ============================================================
# WEIGHTED VOTING — GIVE BETTER MODELS MORE POWER
# ============================================================

# Weights: give the Decision Tree (index 1) more power
weighted_voting = VotingRegressor(
    estimators=[('lr', model1), ('dt', model2), ('svm', model3)],
    weights=[1, 2, 1]
    # U CAN USE A LOOP TO CHECK THE VALUE OF EACH MODEL AND
    # THEN ASSIGN THE WEIGHTS ACCORDINGLY USING CROSS
    # VALIDATION ACCURACY
)


# ============================================================
# FINDING BEST WEIGHTS — BRUTE FORCE WITH LOOP
# ============================================================

lr  = LinearRegression()
dt  = DecisionTreeRegressor()
svr = SVR()

estimators = [('lr', lr), ('dt', dt), ('svr', svr)]

# First check each model individually
for estimator in estimators:
    scores = cross_val_score(estimator[1], X, y,
                             scoring='r2', cv=10)
    print(estimator[0], np.round(np.mean(scores), 2))

# Then check the ensemble
from sklearn.ensemble import VotingRegressor
vr = VotingRegressor(estimators)
scores = cross_val_score(vr, X, y, scoring='r2', cv=10)
print("Voting Regressor", np.round(np.mean(scores), 2))
# Compare this number to the individual models above —
# if ensemble score is higher, the voting is helping!

# Brute force search for best weight combination
for i in range(1, 4):
    for j in range(1, 4):
        for k in range(1, 4):
            vr = VotingRegressor(estimators,
                                  weights=[i, j, k])
            scores = cross_val_score(vr, X, y,
                                     scoring='r2', cv=10)
            print("For i={},j={},k={}".format(i, j, k),
                  np.round(np.mean(scores), 2))
# Tries all 27 combinations of weights (1,2,3 for each)
# and prints R2 score for each — pick the highest one


# ============================================================
# SAME ALGO BUT DIFFERENT PARAMETERS IN VOTING
#
# U CAN ALSO USE DIFFERENT PARAMETERS OF THE SAME ALGO TO
# TUNE THE VALUES ACCORDINGLY BY PUTTING IT IN A VOTING
# ENSEMBLE.
# ============================================================

from sklearn.svm import SVC
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20,
                            n_informative=15, n_redundant=5,
                            random_state=2)

# Same SVM algo, 5 different polynomial degrees
svm1 = SVC(probability=True, kernel='poly', degree=1)
svm2 = SVC(probability=True, kernel='poly', degree=2)
svm3 = SVC(probability=True, kernel='poly', degree=3)
svm4 = SVC(probability=True, kernel='poly', degree=4)
svm5 = SVC(probability=True, kernel='poly', degree=5)

estimators = [('svm1', svm1), ('svm2', svm2),
              ('svm3', svm3), ('svm4', svm4),
              ('svm5', svm5)]

for estimator in estimators:
    x = cross_val_score(estimator[1], X, y,
                        cv=10, scoring='accuracy')
    print(estimator[0], np.round(np.mean(x), 2))
# Find which degree performs best individually before
# combining — weak models hurt the ensemble!


# ============================================================
# SAME ALGO, DIFFERENT max_depth IN VOTING
# ============================================================

dt1 = DecisionTreeRegressor(max_depth=1)
dt2 = DecisionTreeRegressor(max_depth=3)
dt3 = DecisionTreeRegressor(max_depth=5)
dt4 = DecisionTreeRegressor(max_depth=7)
dt5 = DecisionTreeRegressor(max_depth=None)

estimators = [('dt1', dt1), ('dt2', dt2), ('dt3', dt3),
              ('dt4', dt4), ('dt5', dt5)]

for estimator in estimators:
    scores = cross_val_score(estimator[1], X, y,
                             scoring='r2', cv=10)
    print(estimator[0], np.round(np.mean(scores), 2))
# Shallow trees underfit, deep trees overfit — ensemble
# of all depths often beats any single depth alone!


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Voting Ensemble = committee of models, each votes,
#    final answer by majority (hard) or average prob (soft)
#
# 2. DIVERSITY IS THE KEY: models must use different math
#    and each must be better than random (>50% accuracy)
#
# 3. Hard Voting → each model gives a CLASS, majority wins
#    Soft Voting → each model gives PROBABILITY, average
#    wins (usually better, uses confidence info)
#
# 4. Weighted voting → give better models more say
#    Use cross_val_score loop to find best weights
#
# 5. Can use SAME algo with DIFFERENT hyperparameters
#    (different max_depth, different SVM degree) and still
#    get ensemble benefits from their diversity
#
# 6. DISADVANTAGE: "Black Box" — harder to explain than a
#    single model since 3 different mathematical approaches
#    are combined
#
# 7. INTERVIEW TRAP:
#    Q: "Why is Soft Voting usually better than Hard Voting?"
#    Right answer: "Hard Voting treats all model votes
#    equally regardless of confidence. Soft Voting averages
#    the actual probabilities — so a model that's 90% sure
#    has more influence than a model that's 51% sure. This
#    confidence-weighting makes Soft Voting more informative
#    and usually more accurate."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Bagging — Random Forest
#    Training hundreds of trees on different random samples
#    of data, combining their predictions to reduce variance.
# ============================================================ 