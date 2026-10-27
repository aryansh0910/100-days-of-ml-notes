# ============================================================
# ADABOOST — ADAPTIVE BOOSTING
# ============================================================

# ============================================================
# WHAT IS IT?
#
# ALL THE BOOSTING TECHNIQUES ARE BASED ON THE IDEA OF
# "LEARNING FROM MISTAKES." THEY TRAIN MODELS SEQUENTIALLY,
# WHERE EACH MODEL FOCUSES ON THE ERRORS OF THE PREVIOUS ONE.
#
# AND THE FIRST MODEL FROM WHERE IT ALL STARTED IS CALLED
# ADABOOST, WHICH STANDS FOR ADAPTIVE BOOSTING. IT WAS THE
# FIRST BOOSTING ALGORITHM AND IS STILL WIDELY USED TODAY.
#
# AdaBoost, the "Grandfather" of all Boosting techniques.
#
# If Bagging (Random Forest) is about INDEPENDENCE (training
# models in parallel), Boosting is about TEAMWORK (training
# models in sequence).
# ============================================================


# ============================================================
# KEY TERMS
#
# WEAK LEARNER: A model that performs just slightly better
# than random guessing. In AdaBoost, we typically use
# Decision Stumps as weak learners.
#
# DECISION STUMP: A decision tree with ONLY ONE SPLIT.
# It's a "weak learner" that is just slightly better than
# random guessing.
#
# AND IN CASE OF ADABOOST THERE IS NO 0 OR 1 — INSTEAD
# THERE IS 1 AND -1.
# ============================================================


# ============================================================
# 1. THE CORE PHILOSOPHY — "LEARNING FROM MISTAKES"
#
# AdaBoost is like a student who takes a practice exam
# multiple times. After the first attempt, the student
# looks at the questions they got WRONG and spends 90% of
# their time studying only those specific topics before the
# next attempt.
# ============================================================


# ============================================================
# HOW ADABOOST WORKS — STEP BY STEP
#
# AT FIRST U WILL ASSIGN THE WEIGHTS TO EACH ROW TO BE
# 1/N (WHERE N IS THE NUMBER OF ROWS).
#
# THEN U TRAIN A DECISION STUMP ON THIS WEIGHTED DATA.
#
# AFTER THE FIRST STUMP MAKES ITS PREDICTIONS, U UPDATE
# THE WEIGHTS:
# → INCREASE THE WEIGHTS OF THE MISCLASSIFIED ROWS
# → DECREASE THE WEIGHTS OF THE CORRECTLY CLASSIFIED ROWS
#
# AND THEN U CREATE A RANGE ADDING ALL THE WEIGHTS LIKE A
# CUMULATIVE SUM, AND THEN U GENERATE A RANDOM NUMBER
# BETWEEN 0 AND THE TOTAL WEIGHT TO SELECT THE ROWS
# RANDOMLY. THERE WILL BE MORE CHANCES THAT THE RANDOM NO.
# GENERATED WILL FALL INTO THE RANGE OF THE MISCLASSIFIED
# ROWS BECAUSE THEY HAVE HIGHER WEIGHTS. SO THAT ROW WILL
# BE SELECTED MORE OFTEN FOR THE NEXT STUMP TO FOCUS ON.
#
# THEN U TRAIN THE NEXT STUMP ON THIS NEW WEIGHTED DATA,
# AND REPEAT THIS PROCESS FOR A PREDEFINED NUMBER OF
# ITERATIONS OR UNTIL THE MODEL PERFORMS WELL ENOUGH.
#
# VISUAL — the weight update process:
#
# 5 rows, initial weight = 1/5 = 0.2 each
# ┌─────┬────────┬────────┬──────────────┐
# │ Row │ Weight │ Pred   │ New Weight   │
# ├─────┼────────┼────────┼──────────────┤
# │  1  │  0.2   │ ✅ RIGHT│  decreased  │
# │  2  │  0.2   │ ✅ RIGHT│  decreased  │
# │  3  │  0.2   │ ❌ WRONG│  INCREASED  │ ← gets more focus
# │  4  │  0.2   │ ✅ RIGHT│  decreased  │
# │  5  │  0.2   │ ❌ WRONG│  INCREASED  │ ← gets more focus
# └─────┴────────┴────────┴──────────────┘
#
# Cumulative weight range after update:
# Row 1: [0.00 - 0.10]  ← small range, less likely selected
# Row 2: [0.10 - 0.20]  ← small range
# Row 3: [0.20 - 0.45]  ← LARGE range, likely selected ✅
# Row 4: [0.45 - 0.55]  ← small range
# Row 5: [0.55 - 1.00]  ← LARGE range, likely selected ✅
#
# Random number = 0.35 → falls in Row 3's range → Row 3
# selected for next stump's training data. Misclassified
# rows get picked more often — next stump FORCED to fix them!
# ============================================================


# ============================================================
# FINAL VOTE — WEIGHTED COMBINATION
#
# Not all stumps get equal say in the final prediction.
# More accurate stumps get a LOUDER "VOICE."
#
# VISUAL — final vote:
#
# Stump 1 (80% accurate) → weight = HIGH  → says +1 (Class A)
# Stump 2 (55% accurate) → weight = LOW   → says -1 (Class B)
# Stump 3 (75% accurate) → weight = MED   → says +1 (Class A)
#
# Weighted sum = HIGH×(+1) + LOW×(-1) + MED×(+1)
#              = positive → Final prediction = Class A ✅
#
# Compare to Random Forest where ALL trees get equal vote —
# AdaBoost rewards the more confident, accurate stumps.
# ============================================================


# ============================================================
# 4. WHY USE ADABOOST?
#
# PRECISION: Excellent at reducing BIAS (underfitting). It
# forces the model to capture the nuances of the data.
#
# AUTOMATIC FEATURE SELECTION: Since it uses Decision
# Stumps, it naturally identifies the most important
# features very quickly.
#
# EFFICIENCY: Often achieves high accuracy with FEWER trees
# than a Random Forest.
# ============================================================


# ============================================================
# 5. DECISION STUMP vs FULL TREE
#
# IN RANDOM FOREST: we usually let trees grow deep
# (max_depth=None) → each tree is a STRONG learner.
#
# IN ADABOOST: we do the OPPOSITE. We use "WEAK LEARNERS"
# (trees that are barely better than guessing). The magic
# isn't in any single tree; it's in the CUMULATIVE
# CORRECTION of 50 different mistakes.
#
# VISUAL:
#
# Random Forest:
# [Deep Tree] + [Deep Tree] + [Deep Tree] → average
# Each tree is STRONG, errors averaged out
#
# AdaBoost:
# [Stump] → [Stump fixing mistake 1] →
# [Stump fixing mistake 2] → ... → weighted sum
# Each stump is WEAK, but together they fix every mistake
# ============================================================


# ============================================================
# CODE
# ============================================================

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=10000, n_features=10,
                            n_informative=3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ada = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    # max_depth=1 → Decision Stump (one split only)
    n_estimators=100,    # how many stumps to train
    learning_rate=1.0,   # how much each stump contributes
    random_state=42
)

ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)

print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred))


# ============================================================
# BAGGING vs BOOSTING — SIDE BY SIDE
#
# ┌─────────────────┬──────────────────┬───────────────────┐
# │                 │ Bagging          │ Boosting          │
# │                 │ (Random Forest)  │ (AdaBoost)        │
# ├─────────────────┼──────────────────┼───────────────────┤
# │ Training order  │ Parallel         │ Sequential        │
# │ Tree type       │ Deep (strong)    │ Stump (weak)      │
# │ Data weights    │ Equal always     │ Updated each round│
# │ Tree weights    │ Equal vote       │ Accurate=more say │
# │ Goal            │ Reduce Variance  │ Reduce Bias       │
# │ Risk            │ None (stable)    │ Sensitive to      │
# │                 │                  │ outliers (outliers │
# │                 │                  │ get high weights) │
# └─────────────────┴──────────────────┴───────────────────┘
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. AdaBoost = first and simplest boosting algorithm
#    Uses Decision Stumps (max_depth=1) as weak learners
#
# 2. Labels are +1 and -1 (not 0 and 1 like most models)
#
# 3. Core mechanism:
#    → All rows start with equal weight (1/N)
#    → Wrong predictions → weight INCREASES
#    → Right predictions → weight DECREASES
#    → Next stump trained on re-weighted data
#    → Repeat until n_estimators stumps are trained
#
# 4. Selection using cumulative weight range:
#    Misclassified rows have larger ranges → random number
#    more likely to fall in their range → selected more
#
# 5. Final prediction = WEIGHTED vote of all stumps
#    More accurate stumps get more say (unlike RF)
#
# 6. AdaBoost reduces BIAS (underfitting)
#    Random Forest reduces VARIANCE (overfitting)
#
# 7. INTERVIEW TRAP:
#    Q: "What is the weakness of AdaBoost?"
#    Right answer: "AdaBoost is very sensitive to OUTLIERS.
#    When a row is misclassified, its weight keeps getting
#    increased. If that row is an outlier (a genuinely noisy
#    data point that shouldn't be learned from), AdaBoost
#    will keep giving it higher and higher weight, forcing
#    all subsequent stumps to focus on that outlier. This
#    can badly hurt performance — unlike Random Forest which
#    dilutes outlier influence by averaging across trees."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Gradient Boosting
#    More powerful than AdaBoost — instead of updating
#    weights, it fits each new tree on the RESIDUAL ERRORS
#    of the previous model.
# ============================================================