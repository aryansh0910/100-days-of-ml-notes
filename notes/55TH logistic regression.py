# ============================================================
# LOGISTIC REGRESSION
# ============================================================

# ============================================================
# WHAT IS IT?
#
# LOGISTIC REGRESSION IS USED FOR CLASSIFICATION PROBLEMS
# LIKE 0 AND 1.
# ============================================================


# ============================================================
# 1. THE GEOMETRIC PROBLEM
#
# If you use standard Linear Regression to predict Churn
# (0 or 1), your "best-fit line" will eventually go ABOVE 1
# and BELOW 0. But a probability can never be outside the
# 0 to 1 range!
#
# VISUAL — why Linear Regression breaks for classification:
#
#  y
#  1.5 |                          ●
#    1 |─────────────────●●●●●────────
#      |              ●●
#  0.5 |          ●●          ← straight line keeps going
#      |      ●●                 past 0 and 1, which makes
#    0 |──●●──────────────────── no sense for probability!
# -0.5 |●
#      |________________________ x
#
# We need a curve that's BOUNDED between 0 and 1.
# ============================================================


# ============================================================
# THE SIGMOID FUNCTION — THE FIX
#
# We take the linear equation z = mx + b and pass it
# through the Sigmoid (Logistic) Function:
#
# ┌─────────────────────────────────────────────┐
# │                                             │
# │              1                              │
# │  σ(z)  =  ─────────                         │
# │            1 + e^(-z)                       │
# │                                             │
# └─────────────────────────────────────────────┘
#
# BEHAVIOR:
# When z is a HUGE POSITIVE number → σ(z) close to 1
# When z is a HUGE NEGATIVE number → σ(z) close to 0
# When z is 0                      → σ(z) exactly 0.5
#
# VISUAL — the S-curve:
#
#  Probability
#    1 |              ________●●●●
#      |          ●●●●
#  0.5 |       ●●●
#      |    ●●●
#    0 |●●●●________________________
#      |________________________ z (linear output)
#     -∞    -2  0   2           +∞
#
# Notice: no matter how extreme z gets, the output is
# squashed between 0 and 1 — perfect for probability!
# ============================================================


# ============================================================
# 3. DECISION BOUNDARY
#
# Once the model gives you a probability (e.g. "This
# customer has a 0.72 chance of churning"), you need to
# make a final call. We usually use a THRESHOLD of 0.5:
#
# Prob ≥ 0.5 → Predict Churn (1)
# Prob < 0.5 → Predict Stay (0)
#
# Geometrically, this creates a DECISION BOUNDARY — a line
# (or plane) that separates the "Will Churn" dots from the
# "Won't Churn" dots.
#
# VISUAL:
#
#  Feature 2
#    |  ●  ●     ●  ●    ← Will Churn (1) region
#    |    ●   ╲    ●
#    |  ●    ╲   ●        ← decision boundary line
#    |    ○  ╲ ○          (where probability = 0.5)
#    |  ○   ○ ╲    ○
#    |    ○    ╲  ○      ← Won't Churn (0) region
#    |________________________ Feature 1
# ============================================================


# ============================================================
# 4. THE COST FUNCTION — LOG LOSS
#
# In Linear Regression, we used MSE (Mean Squared Error).
# But squaring the errors of a Sigmoid curve creates a
# "wavy" surface with many fake valleys, making Gradient
# Descent FAIL (it gets stuck in local minima).
#
# Instead, we use LOG LOSS (Binary Cross-Entropy).
#
# THE LOGIC:
# If the true label is 1 and you predict 0.99 → penalty
# is TINY
# If the true label is 1 and you predict 0.01 → penalty
# is MASSIVE (approaching infinity)
#
# This forces the model to be very "confident" when it's
# right, and HEAVILY punishes confident wrong predictions.
#
# VISUAL — Log Loss penalty curve (true label = 1):
#
# Predicted prob → Penalty
# 0.99 → tiny penalty   (●  close to bottom of curve)
# 0.50 → moderate       (   ●)
# 0.10 → large          (        ●)
# 0.01 → massive        (              ● → almost ∞)
#
#  Penalty
#    |                                    ●
#    |                              ●
#    |                       ●
#    |              ●
#    |     ●●●●
#    |________________________________
#    0    0.25   0.5   0.75    1.0
#         Predicted probability (true label = 1)
#
# WHY NOT MSE?
# MSE on a sigmoid output creates a non-convex (wavy)
# cost surface — gradient descent can get trapped in fake
# local minima. Log Loss creates a smooth, convex bowl —
# gradient descent reliably finds the true minimum.
# ============================================================


# ============================================================
# CODE
# ============================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample churn-style data
np.random.seed(42)
n = 200
df = pd.DataFrame({
    'Monthly_Usage': np.random.randint(10, 100, n),
    'Tenure_Months' : np.random.randint(1, 60, n),
})
# Create a target where higher usage + lower tenure = more likely to churn
df['Churn'] = ((df['Monthly_Usage'] > 60) &
               (df['Tenure_Months'] < 20)).astype(int)

X = df[['Monthly_Usage', 'Tenure_Months']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict probabilities (not just 0/1)
probabilities = model.predict_proba(X_test_scaled)
print("Probabilities (class 0, class 1):")
print(probabilities[:5])

# Predict actual classes using 0.5 threshold internally
predictions = model.predict(X_test_scaled)
print("\nPredictions:", predictions[:5])


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Linear Regression's line goes beyond 0-1, useless for
#    probability. Sigmoid squashes any z value into 0-1.
#
# 2. Sigmoid formula: σ(z) = 1 / (1 + e^(-z))
#    z=0 → 0.5, z→+∞ → 1, z→-∞ → 0
#
# 3. Decision boundary uses threshold 0.5 by default:
#    prob ≥ 0.5 → class 1, prob < 0.5 → class 0
#
# 4. MSE doesn't work for Logistic Regression — creates a
#    non-convex cost surface with fake local minima
#
# 5. Log Loss (Binary Cross-Entropy) is used instead —
#    convex surface, and HEAVILY punishes confident wrong
#    predictions (predicting 0.01 when true label is 1)
#
# 6. INTERVIEW TRAP:
#    Q: "Why can't we use MSE as the cost function for
#        Logistic Regression?"
#    Right answer: "Squaring the sigmoid's errors creates a
#    non-convex cost surface with many local minima, so
#    gradient descent can get stuck and never find the true
#    best solution. Log Loss is convex — it has exactly one
#    global minimum, which gradient descent can reliably find."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Logistic Regression — Multiclass Classification
#    Extending binary classification to handle more than
#    two classes using One-vs-Rest and Softmax.
# ============================================================