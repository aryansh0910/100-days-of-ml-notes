# ============================================================
# ROC CURVE AND AUC SCORE
# ============================================================

# ============================================================
# WHAT IS IT?
#
# The ROC Curve (Receiver Operating Characteristic) is the
# ultimate "TRUTH DETECTOR" for classification models. It
# solves a major problem: Accuracy only tells you IF the
# model is right, but the ROC Curve tells you HOW CONFIDENT
# and CAPABLE the model is at separating two classes
# (like "Sick" vs. "Healthy").
#
# THE MAIN REASON U USE ROC CURVE IS THE THRESHOLD
# SELECTION PROBLEM. IF WE SET THE THRESHOLD TO 0.80 THEN
# BELOW IT THE PREDICTION WILL BE 0 AND ABOVE IT THE
# PREDICTION WILL BE 1. GENERALLY IT IS SET TO 0.5 SO U
# CAN CONTROL IT. AND THE ROC CURVE HELPS U TO DECIDE WHAT
# SHOULD THAT THRESHOLD BE.
#
# THE ROC CURVE SHOWS YOU HOW YOUR MODEL PERFORMS ACROSS
# ALL POSSIBLE THRESHOLDS, SO YOU CAN PICK THE ONE THAT
# GIVES YOU THE BEST BALANCE BETWEEN CATCHING TRUE
# POSITIVES AND AVOIDING FALSE POSITIVES.
# ============================================================


# ============================================================
# THE TWO KEY METRICS
#
# TPR (True Positive Rate) = TP / (TP + FN)
# → How many actual positives did we catch?
# ITS LIKE A BENEFIT OF THE MODEL — HOW MUCH YOUR MODEL
# DID A GOOD JOB IN CATCHING THE TRUE POSITIVES.
# LIKE 80% OF THE ACTUAL SICK PEOPLE WERE CORRECTLY
# IDENTIFIED AS SICK BY THE MODEL.
# ALSO CALLED RECALL OR SENSITIVITY.
# THE HIGHER THE TPR, THE BETTER. → Y-AXIS
#
# FPR (False Positive Rate) = FP / (FP + TN)
# → How many actual negatives did we accidentally label
# as positives?
# ITS LIKE A COST OF THE MODEL — HOW MUCH YOUR MODEL DID
# A BAD JOB IN CATCHING THE FALSE POSITIVES.
# LIKE 10% OF THE ACTUAL HEALTHY PEOPLE WERE INCORRECTLY
# IDENTIFIED AS SICK BY THE MODEL.
# ALSO CALLED (1 - SPECIFICITY).
# THE LOWER THE FPR, THE BETTER. → X-AXIS
#
# U CAN SAY ITS A GRAPH B/W THE BENEFIT (TPR) AND THE
# COST (FPR) OF YOUR MODEL.
#
# THE ROC CURVE PLOTS TPR (Y-AXIS) AGAINST FPR (X-AXIS)
# AS YOU VARY THE THRESHOLD.
# ============================================================


# ============================================================
# 2. HOW TO READ THE CURVE
#
# VISUAL — ROC Curve shapes:
#
# TPR
# 1.0|●────────────────  ← Perfect model (AUC=1.0)
#    |  ●──────────      ← Good model (AUC=0.85)
#    |     ●──────       ← OK model (AUC=0.70)
# 0.5|        ╱ ←── Random guess (AUC=0.5, diagonal)
#    |      ╱
# 0.0|────╱────────────
#    0.0  0.5  1.0  FPR
#
# TOP-LEFT CORNER → perfect (high TPR, low FPR)
# DIAGONAL LINE   → random guessing (useless)
# BELOW DIAGONAL  → worse than random (something wrong!)
#
# PERFECT MODEL: Curve HUGS the top-left corner.
# Catches everyone sick without accusing healthy people.
#
# RANDOM MODEL: A diagonal line from bottom-left to
# top-right. This model is basically flipping a coin.
#
# BAD MODEL: Curve is BELOW the diagonal line.
# Worse than a coin flip!
# ============================================================


# ============================================================
# 3. THE AUC SCORE (AREA UNDER THE CURVE)
#
# The AUC is the SINGLE NUMBER that summarizes the ROC curve.
#
# AUC = 1.0 → Perfect separation
# AUC = 0.5 → No better than guessing
# AUC > 0.8 → Generally considered a very strong model
#
# ON THE BASIS OF THE AUC VALUE U CAN JUDGE HOW GOOD YOUR
# MODEL IS. THE MORE THE AUC VALUE THE BETTER YOUR MODEL
# IS AT SEPARATING THE CLASSES.
# U CAN ALSO COMPARE THE AUC VALUES OF DIFFERENT MODELS
# TO SEE WHICH ONE IS BETTER. THE MODEL WITH THE HIGHER
# AUC VALUE IS THE BETTER MODEL.
# ============================================================


# ============================================================
# 5. WHY IS THIS BETTER THAN ACCURACY?
#
# Imagine a model for a rare disease:
#
# Model A: Predicts "Healthy" for everyone.
# Accuracy = 99% ← looks great
# AUC = 0.5      ← USELESS (never catches sick people)
#
# Model B: Makes some mistakes but identifies sick correctly.
# Accuracy = 90% ← looks worse
# AUC = 0.9      ← EXCELLENT (actually useful!)
#
# The ROC curve proves that Model B is actually SUPERIOR
# even though its "Accuracy" is lower. ✅
# ============================================================


# ============================================================
# CODE — PLOTTING THE ROC CURVE
# ============================================================

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 1. Get the probabilities for the 'Positive' class (col 1)
# NOTE: SVM needs probability=True
#       XGBoost does this by default
y_probs = model.predict_proba(X_test)[:, 1]
# [:, 1] → second column = probability of class 1 (positive)

# 2. Calculate the curve points
# roc_curve returns FPR, TPR, and thresholds for every
# possible cutoff point
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# 3. Calculate the AUC Score
auc_value = roc_auc_score(y_test, y_probs)

# 4. Plot the results
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Model AUC: {auc_value:.4f}")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
# k-- = black dashed diagonal line (the random baseline)
plt.xlabel("False Positive Rate (Cost)")
plt.ylabel("True Positive Rate (Benefit)")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("AUC Score:", auc_value)


# ============================================================
# FINDING THE BEST THRESHOLD — YOUDEN'S J
#
# The default threshold of 0.5 is NOT always optimal.
# Youden's J statistic finds the threshold that gives the
# BEST BALANCE between TPR and FPR.
#
# Formula: J = TPR - FPR
# The threshold with the HIGHEST J score is the best one.
# ============================================================

import numpy as np

# 1. Get the probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# 2. Get the curve data
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# 3. Calculate Youden's J for every threshold
j_scores = tpr - fpr

# 4. Find the index of the highest J score
best_index     = np.argmax(j_scores)
best_threshold = thresholds[best_index]

print("Best Threshold:", best_threshold)
print("Best TPR (Recall):", tpr[best_index])
print("Best FPR (False Alarm):", fpr[best_index])

# VISUAL — Youden's J on the curve:
#
# TPR
# 1.0|    ●──────────
#    |  ●      ↑ best threshold point (max J = TPR-FPR)
#    |●              The vertical distance between the
#    |╱              curve and the diagonal is maximized
# 0.0|────────────── FPR
#
# The point where the curve is FARTHEST from the diagonal
# = best threshold = highest Youden's J


# ============================================================
# USING THE BEST THRESHOLD IN YOUR MODEL
#
# HOW TO USE THAT THRESHOLD INTO YOUR CODE FOR ANY MODEL
# ============================================================

# Step 1: Get the raw probabilities (0.0 to 1.0)
y_probs = model.predict_proba(X_test)[:, 1]

# Step 2: Set your custom threshold
my_threshold = 0.32
# (the one found using Youden's J above)

# Step 3: Manually create predictions
# "If prob >= 0.32, predict 1. Otherwise predict 0."
custom_preds = (y_probs >= my_threshold).astype(int)

# Step 4: Evaluate with custom predictions
from sklearn.metrics import classification_report
print(classification_report(y_test, custom_preds))
# Compare this with default threshold (0.5) results
# to see if the custom threshold improved recall


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. ROC Curve = graph of TPR (benefit) vs FPR (cost)
#    across ALL possible thresholds
#
# 2. TPR = TP/(TP+FN) = Recall — how many sick caught?
#    FPR = FP/(FP+TN) — how many healthy accused wrongly?
#
# 3. Curve hugging top-left = great model
#    Curve near diagonal    = random guessing (useless)
#    Curve below diagonal   = worse than random
#
# 4. AUC = single number summarizing the curve
#    AUC=1.0 → perfect, AUC=0.5 → random, AUC>0.8 → strong
#
# 5. Use Youden's J (TPR - FPR) to find the optimal
#    threshold — not always 0.5!
#
# 6. model.predict_proba()[:, 1] gives probabilities
#    needed to plot the ROC curve — you can't use
#    predict() (which gives 0/1 only)
#
# 7. INTERVIEW TRAP:
#    Q: "Why is AUC-ROC better than accuracy for
#        imbalanced datasets?"
#    Right answer: "Accuracy is dominated by the majority
#    class — a model predicting 'Healthy' for everyone
#    gets 99% accuracy on a rare disease dataset but
#    catches zero sick people. AUC-ROC measures the
#    model's ability to RANK positives higher than
#    negatives across ALL thresholds, so it correctly
#    gives this useless model an AUC of 0.5. AUC is
#    also threshold-independent — it tells you the
#    model's fundamental separating power, then you
#    separately choose the best threshold using
#    Youden's J based on business priorities."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Model Deployment with Streamlit
#    Saving your model and building a web app interface
#    for your Resume Screener project.
# ============================================================