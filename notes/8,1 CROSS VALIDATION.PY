# ============================================================
# CROSS VALIDATION
# ============================================================

# ============================================================
# WHAT IS IT?
# Cross Validation = a technique to evaluate your model
# MORE HONESTLY by testing it on multiple different splits
# of your data instead of just one single train/test split.
# ============================================================


# ============================================================
# THE PROBLEM WITH NORMAL TRAIN/TEST SPLIT
# ============================================================
#
# When we do a normal split:
# → X_train, X_test = train_test_split(X, y, test_size=0.2)
#
# We are ONLY testing on 20% of the data — ONE TIME.
# What if that 20% happened to be "easy" data by luck?
# What if it happened to be "hard" data by bad luck?
#
# ANALOGY:
# Imagine judging a cricket player's skill based on
# just ONE match. What if that match was against a
# very weak team? Or a very strong team?
# ONE match is not enough to judge true skill.
# You need to see them play MULTIPLE matches.
#
# Cross Validation = making the model play multiple matches.
# ============================================================


# ============================================================
# HOW K-FOLD CROSS VALIDATION WORKS
# ============================================================
#
# STEP 1: Split the entire dataset into K equal parts (folds)
# STEP 2: Train on K-1 folds, Test on the remaining 1 fold
# STEP 3: Repeat K times — each fold gets a turn as test set
# STEP 4: Average all K scores → final honest score
#
# EXAMPLE WITH K=5:
#
# Fold 1: [TEST] [train] [train] [train] [train] → Score 1
# Fold 2: [train] [TEST] [train] [train] [train] → Score 2
# Fold 3: [train] [train] [TEST] [train] [train] → Score 3
# Fold 4: [train] [train] [train] [TEST] [train] → Score 4
# Fold 5: [train] [train] [train] [train] [TEST] → Score 5
#
# Final Score = Average of Score 1,2,3,4,5
#
# ANALOGY:
# Like judging a cricket player across 5 different matches
# against 5 different teams → now you get a TRUE picture
# of their actual skill level!
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# ============================================================
# STEP 1 — PREPARE DATA
# ============================================================

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# NOTE: We do NOT split into train/test here manually
# Cross validation handles the splitting for us internally!


# ============================================================
# STEP 2 — DEFINE MODEL
# ============================================================

model = RandomForestClassifier(n_estimators=100, random_state=42)


# ============================================================
# STEP 3 — APPLY CROSS VALIDATION
# ============================================================

# cv=5 means K=5 → 5 folds → model trains and tests 5 times
# scoring='accuracy' → metric we want to measure
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("=" * 55)
print("K-FOLD CROSS VALIDATION RESULTS (K=5)")
print("=" * 55)
print(f"Score each fold : {scores}")
print(f"Mean Accuracy   : {scores.mean():.4f}")
print(f"Std Deviation   : {scores.std():.4f}")
print()

# HOW TO READ THE OUTPUT:
#
# Score each fold : [0.92, 0.89, 0.91, 0.93, 0.90]
# Mean Accuracy   : 0.9100   ← this is your REAL model score
# Std Deviation   : 0.0140   ← how consistent your model is
#
# Low Std Dev → model performs consistently → GOOD ✅
# High Std Dev → model is unstable across folds → BAD ❌


# ============================================================
# STEP 4 — USING KFold MANUALLY (more control)
# ============================================================

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 55)
print("MANUAL K-FOLD (with fold details)")
print("=" * 55)

fold_num = 1
all_scores = []

for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    all_scores.append(score)

    print(f"Fold {fold_num} → Test Score: {score:.4f}")
    fold_num += 1

print(f"\nFinal Mean Score : {np.mean(all_scores):.4f}")
print(f"Std Deviation    : {np.std(all_scores):.4f}")


# ============================================================
# NORMAL SPLIT vs CROSS VALIDATION — WHEN TO USE WHAT
# ============================================================
#
# Normal Train/Test Split:
# → Fast, simple, good for quick checks
# → Use when dataset is very large (100k+ rows)
# → Use during initial experimentation
#
# Cross Validation:
# → Slower but way more honest and reliable
# → Use when dataset is small or medium sized
# → Use before finalizing your model
# → Use in Kaggle competitions for reliable scoring
# → ALWAYS use when comparing two different models
#
# RULE OF THUMB:
# Small dataset (< 10k rows)  → Always use Cross Validation
# Large dataset (> 100k rows) → Normal split is usually fine
# ============================================================


# ============================================================
# COMMON VALUES OF K
# ============================================================
#
# K=5  → Most commonly used, good balance of speed vs accuracy
# K=10 → More reliable, slightly slower
# K=3  → Use when dataset is very small and training is slow
#
# Special case: K = n (number of rows) = Leave One Out (LOO)
# → Most accurate but extremely slow
# → Rarely used in practice
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Normal train/test split evaluates model only ONCE
#    → Can be misleading depending on which rows end up
#    in test set (luck factor)
#
# 2. Cross Validation evaluates model K times on K different
#    splits → gives a much more honest and reliable score
#
# 3. Final score = MEAN of all K fold scores
#    Consistency = STD DEVIATION of all K fold scores
#
# 4. Low Std Dev = consistent model = good ✅
#    High Std Dev = unstable model = investigate ❌
#
# 5. cross_val_score() does everything automatically
#    Just pass model, X, y, cv=5
#
# 6. Always use Cross Validation when:
#    → Comparing two models
#    → Small/medium dataset
#    → Before final model submission
#
# 7. Cross Validation does NOT replace the final test set
#    → Still keep a separate holdout test set for final eval
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Feature Engineering
#    The process of creating new meaningful features from
#    existing ones to help the model learn better patterns.
#    Example: from "Date of Purchase" extracting
#    "Day of Week", "Month", "Is Weekend" etc.
# ============================================================
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler    

X= make_classification(n_samples=1000, n_features=10, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = make_classification(n_samples=1000, n_features=10, random_state=42)[1]
model = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=15)# with overfitting
model = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=5)# without overfitting
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(scores.mean())
print(scores.std())
