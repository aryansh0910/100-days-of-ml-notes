# ============================================================
# HANDLING IMBALANCED DATA
# ============================================================

# ============================================================
# WHAT IS IT?
#
# Dealing with Imbalanced Data is one of the most common
# challenges in real-world Machine Learning.
#
# PROBLEMS WITH IMBALANCED DATA:
#
# BIASED LEARNING: The model becomes biased towards the
# majority class and fails to learn the patterns of the
# minority class.
#
# MISLEADING ACCURACY: A model can achieve high accuracy
# by simply predicting the majority class.
#
# VISUAL — the accuracy trap:
#
# Fraud dataset: 9990 Normal, 10 Fraud
# Dumb model predicts "Normal" for EVERYONE:
# Accuracy = 9990/10000 = 99.9% ← looks amazing ✅
# But caught 0 fraud cases ❌ completely useless!
#
# This is why accuracy is a LIE on imbalanced data.
# Use Precision, Recall, F1-Score instead.
# ============================================================


# ============================================================
# THREE STRATEGIES TO HANDLE IMBALANCED DATA
#
# STRATEGY 1 → RESAMPLING (Fix the Data)
# STRATEGY 2 → ALGORITHMIC FIXES (Fix the Model)
# STRATEGY 3 → BETTER METRICS (Fix how you Evaluate)
# ============================================================


# ============================================================
# STRATEGY 1A — UNDERSAMPLING
#
# REDUCE the number of samples in the majority class to
# match the minority class.
#
# ITS LIKE SELECTING A RANDOM SUBSET (USUALLY EQUAL TO
# THE NO. OF THE MINORITY CLASS) OF THE MAJORITY CLASS
# TO TRAIN THE MODEL.
#
# RISK: You might LOSE IMPORTANT INFORMATION from the
# majority class.
#
# FOR THIS U HAVE TO USE IMBLEARN LIBRARY (very famous
# library for handling imbalanced data)
#
# VISUAL:
#
# BEFORE:                     AFTER:
# Normal: ●●●●●●●●●● (1000)   Normal: ●●● (100)
# Fraud:  ×× (100)             Fraud:  ×× (100)
#                              ← balanced but lost 900 rows!
# ============================================================

from imblearn.under_sampling import RandomUnderSampler

# 1. Initialize the undersampler
# sampling_strategy=1.0 makes the classes exactly 50/50
rus = RandomUnderSampler(sampling_strategy=1.0,
                          random_state=42)

# 2. Resample the TRAINING data ONLY
X_train_under, y_train_under = rus.fit_resample(
    X_train, y_train
)

# 3. Verify the new counts
print(y_train_under.value_counts())


# ============================================================
# STRATEGY 1B — OVERSAMPLING (Random)
#
# INCREASE the number of samples in the minority class
# by DUPLICATING existing minority class data points until
# it matches the majority class.
#
# RISK: Can lead to OVERFITTING because you're just
# repeating the same points — model memorizes them.
# ============================================================

from imblearn.over_sampling import RandomOverSampler

# sampling_strategy=1.0 → makes classes exactly 50/50
ros = RandomOverSampler(sampling_strategy=1.0,
                         random_state=42)

X_train_over, y_train_over = ros.fit_resample(
    X_train, y_train
)

print(y_train_over.value_counts())


# ============================================================
# STRATEGY 1C — SMOTE (Best Oversampling Technique)
#
# SMOTE (Synthetic Minority Over-sampling Technique):
# Instead of just DUPLICATING, SMOTE CREATES NEW SYNTHETIC
# data points for the minority class.
#
# FIRST THE ALGO FINDS THE K NEAREST NEIGHBORS FOR EACH
# MINORITY CLASS POINT, THEN IT RANDOMLY SELECTS ONE OF
# THOSE NEIGHBORS AND CREATES A NEW POINT SOMEWHERE ALONG
# THE LINE BETWEEN THEM.
#
# RISK: Can create "unrealistic" data points if noise is
# present.
#
# VISUAL — SMOTE vs Random Oversampling:
#
# Random Oversampling:
# ×  ×  ×  ×(copy) ×(copy) ×(copy)
# (just duplicates → overfitting risk)
#
# SMOTE:
# ×    ×    ×new  ×new  ×new  ×new
#      ↑         ↑
#   original   synthetic (new point between two originals)
# (creates realistic new points → better generalization)
# ============================================================

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=1.0, random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train, y_train
)

print(y_train_smote.value_counts())


# ============================================================
# STRATEGY 2 — ALGORITHMIC FIXES (Fix the Model)
#
# ENSEMBLE METHODS (ONE OF THE BEST TECHNIQUES):
# Instead of changing the data, tell the model:
# "The rare class is 10x more important. If you miss a
# Fraud case, I will PUNISH you 10 times harder."
#
# SVM          → class_weight='balanced'
# Random Forest → class_weight='balanced'
# XGBoost      → scale_pos_weight parameter
#
# FORMULA for XGBoost:
# scale_pos_weight = count(negative) / count(positive)
# ============================================================

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Example: 990 healthy, 10 sick
ratio = 990 / 10  # = 99

# XGBoost — scale_pos_weight
xgb_balanced = XGBClassifier(scale_pos_weight=ratio)
xgb_balanced.fit(X_train, y_train)

# Random Forest — class_weight='balanced'
rf_balanced = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=100
)

# SVM — class_weight='balanced'
svm_balanced = SVC(class_weight='balanced', kernel='rbf')


# ============================================================
# STRATEGY 3 — BETTER METRICS
#
# 1. THE "ACCURACY TRAP":
# When data is imbalanced, Accuracy is a LIE.
#
# PRECISION: "Of all the times I predicted Fraud, how
# many were actually Fraud?"
#
# RECALL (Sensitivity): "Of all the actual Fraud cases,
# how many did I successfully catch?"
#
# F1-SCORE: The balance between Precision and Recall.
#
# Always look at the CONFUSION MATRIX and F1-SCORE for
# imbalanced data, not accuracy!
# ============================================================


# ============================================================
# COMPARISON TABLE
#
# ┌──────────────────┬──────────────────┬───────────────────┐
# │ Method           │ Pros             │ Cons              │
# ├──────────────────┼──────────────────┼───────────────────┤
# │ Undersampling    │ Fast, makes      │ Lose valuable     │
# │                  │ training quick   │ majority class    │
# │                  │                  │ data              │
# │ SMOTE            │ No data loss,    │ Can create        │
# │                  │ model sees rare  │ "unrealistic"     │
# │                  │ class better     │ points if noise   │
# │ Class Weights    │ Easiest to       │ Only works if     │
# │                  │ implement, no    │ model supports    │
# │                  │ data change      │ weight params     │
# └──────────────────┴──────────────────┴───────────────────┘
# ============================================================


# ============================================================
# 5. THE GOLDEN RULE OF TESTING
#
# NEVER, EVER RESAMPLE YOUR TEST SET.
#
# If you use SMOTE on your test set, you are testing your
# model on "FAKE" data. Your test set must ALWAYS remain
# a reflection of the REAL WORLD, even if it is 99%
# imbalanced.
#
# CORRECT WORKFLOW:
# ┌──────────────────────────────────────────────┐
# │ 1. Split into train and test FIRST           │
# │ 2. Apply SMOTE/undersampling on TRAIN ONLY   │
# │ 3. Keep test set UNTOUCHED (real world data) │
# │ 4. Evaluate on UNTOUCHED test set            │
# └──────────────────────────────────────────────┘
# ============================================================


# ============================================================
# SUMMARY — WHAT TO DO STEP BY STEP
#
# 1. Check your class counts:
#    If one class < 10-20% of other → imbalanced
#
# 2. Stop looking at "Accuracy" → look at Confusion
#    Matrix and F1-Score
#
# 3. Try scale_pos_weight in XGBoost FIRST — it's usually
#    the cleanest fix (no data modification needed)
#
# 4. If that's not enough → try SMOTE on training data
#
# 5. NEVER resample the test set
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Imbalanced data → accuracy is misleading. Always use
#    Precision, Recall, F1-Score, and Confusion Matrix
#
# 2. Three approaches:
#    Undersampling → remove majority class rows (fast,
#                    but loses information)
#    SMOTE         → create synthetic minority rows
#                    (better than random oversampling)
#    Class weights → tell model to penalize minority class
#                    misses more heavily (cleanest fix)
#
# 3. SMOTE creates NEW points between existing minority
#    points along a line — not just duplicates
#
# 4. XGBoost: scale_pos_weight = count(neg)/count(pos)
#    RF/SVM: class_weight='balanced'
#
# 5. GOLDEN RULE: resample TRAIN only, NEVER the test set
#
# 6. INTERVIEW TRAP:
#    Q: "Why is SMOTE better than random oversampling?"
#    Right answer: "Random oversampling just DUPLICATES
#    existing minority class points — the model sees the
#    exact same rows multiple times and memorizes them,
#    leading to overfitting. SMOTE creates SYNTHETIC new
#    points by interpolating between existing minority
#    samples using their K nearest neighbors. This gives
#    the model new, slightly different examples of the
#    minority class to learn from, reducing overfitting
#    while still balancing the dataset."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Model Deployment with Streamlit
#    Saving your model and building a web interface for
#    your Resume Screener and Plant Disease Detector.
# ============================================================