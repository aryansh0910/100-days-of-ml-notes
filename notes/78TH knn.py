# ============================================================
# K-NEAREST NEIGHBORS (KNN)
# ============================================================

# ============================================================
# WHAT IS IT?
#
# KNN is one of the most intuitive and "human-like"
# algorithms in machine learning. While other models try
# to find a complex mathematical formula or a tree
# structure, KNN follows a simple rule:
# "Tell me who your NEIGHBORS are, and I'll tell you
# who YOU are."
# ============================================================


# ============================================================
# 1. THE CORE PHILOSOPHY
#
# KNN is known as a LAZY LEARNER. It doesn't actually
# "learn" or build a model during the training phase —
# it simply MEMORIZES the entire dataset.
#
# When you ask it to make a prediction for a new point:
#
# STEP 1 — CALCULATE DISTANCE:
# Measure the distance between the new point and EVERY
# other point in the training data.
#
# STEP 2 — FIND NEIGHBORS:
# Identify the K points that are CLOSEST to the new point.
#
# STEP 3 — VOTE/AVERAGE:
# For CLASSIFICATION → MAJORITY VOTE
# (4 neighbors are "Red" and 1 is "Blue" → predict "Red")
# For REGRESSION → AVERAGE value of K neighbors
#
# VISUAL — KNN classification (K=3):
#
#   ●(Churn)  ○(Stay)
#       ●(Churn)   ★ ← new point, what class?
#   ○(Stay)      ●(Churn)
#
# Find 3 nearest to ★:
# Neighbor 1: ●(Churn) — distance 1.2
# Neighbor 2: ●(Churn) — distance 1.5
# Neighbor 3: ○(Stay)  — distance 1.8
#
# Vote: 2 Churn vs 1 Stay → predict CHURN ✅
# ============================================================


# ============================================================
# CHOOSING THE RIGHT K
#
# The value of K determines how your model generalizes:
#
# SMALL K (e.g. K=1): Very sensitive, captures every tiny
# detail → HIGH VARIANCE (Overfitting) — reacts to noise
# and outliers.
#
# LARGE K (e.g. K=100): Too "blurry," averages too many
# neighbors, misses local patterns → HIGH BIAS
# (Underfitting).
#
# U GENERALLY SELECT ODD VALUE OF K AND ITS THE SQUARE
# ROOT OF THE NUMBER OF DATA POINTS IN YOUR TRAINING SET.
# (Odd K prevents ties in classification voting)
#
# THE OTHER AND BETTER WAY IS TO USE CROSS VALIDATION TO
# SELECT THE BEST K VALUE. YOU CAN TRY DIFFERENT K VALUES
# AND SEE WHICH ONE GIVES YOU THE BEST ACCURACY ON THE
# VALIDATION SET. U CAN PLOT THE VALUES OF ACCURACY ON
# THE PLOT AS WELL TO SEE WHICH K VALUE IS THE BEST.
#
# VISUAL — K vs Accuracy curve:
#
# Accuracy
#   |         ●●●
#   |       ●●   ●●
#   |     ●●        ●●●●●
#   |  ●●●                ●●●●  ← underfitting
#   |___________________________________ K
#   1   3   5   7   9   11  13
#            ↑
#       best K (peak accuracy)
# ============================================================


# ============================================================
# PROS AND CONS
#
# ┌──────────────────────┬──────────────────────────────────┐
# │ PROS                 │ CONS                             │
# ├──────────────────────┼──────────────────────────────────┤
# │ Simple: extremely    │ SLOW during prediction: must     │
# │ easy to understand   │ calculate distance to every      │
# │ and explain          │ point for every new prediction   │
# │                      │                                  │
# │ No Assumptions:      │ MEMORY INTENSIVE: must keep      │
# │ doesn't care if      │ entire dataset in RAM to make    │
# │ data is linear or    │ predictions                      │
# │ non-linear           │                                  │
# │                      │ SENSITIVE TO OUTLIERS: one bad   │
# │ Versatile: works for │ data point in a small            │
# │ both regression and  │ neighborhood can ruin prediction │
# │ classification       │                                  │
# └──────────────────────┴──────────────────────────────────┘
# ============================================================


# ============================================================
# SCALING IS MANDATORY
#
# ITS VERY IMPORTANT TO SCALE THE DATA BEFORE USING KNN
# BECAUSE IT RELIES ON DISTANCE CALCULATIONS. IF ONE
# FEATURE HAS A MUCH LARGER SCALE THAN OTHERS, IT CAN
# DOMINATE THE DISTANCE AND LEAD TO POOR PREDICTIONS.
#
# VISUAL — unscaled vs scaled distance:
#
# UNSCALED:
# Point A: Age=25,  Salary=50,000
# Point B: Age=26,  Salary=50,001
# Distance = √((1)² + (1)²) = 1.41
# Age and Salary seem equally important
#
# BUT with real ranges (Age 18-80, Salary 20k-150k):
# Point A: Age=25,  Salary=20,000
# Point B: Age=60,  Salary=20,001
# Distance = √((35)² + (1)²) ≈ 35
# Salary barely contributes, Age DOMINATES ❌
#
# AFTER SCALING: both features contribute equally ✅
# ============================================================


# ============================================================
# CODE
# ============================================================

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

# 1. KNN MUST have scaled data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reg)

# 2. Initialize the model
# n_neighbors (K) is the most important setting
# Start with K=5
knn = KNeighborsRegressor(n_neighbors=5)

# 3. "Train" (Memorize) the data
knn.fit(X_scaled, y_reg)

# 4. Predict for a single sample
prediction = knn.predict(X_scaled[:1])
print("KNN Prediction for the first sample:", prediction[0])


# ============================================================
# FINDING BEST K USING CROSS VALIDATION
# ============================================================

k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y_reg,
                             cv=5, scoring='r2')
    cv_scores.append(scores.mean())

# Plot K vs Accuracy
plt.figure(figsize=(10, 5))
plt.plot(k_values, cv_scores, marker='o', color='teal')
plt.xlabel('K Value')
plt.ylabel('R2 Score (CV)')
plt.title('Finding Best K using Cross Validation')
plt.grid(True)
plt.show()

best_k = k_values[np.argmax(cv_scores)]
print(f"Best K: {best_k}")
# Pick the K at the PEAK of this curve


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. KNN = Lazy Learner — no training phase, just memorizes
#    entire dataset. All computation happens at PREDICTION
#    time, not training time
#
# 2. Classification → majority vote of K neighbors
#    Regression     → average value of K neighbors
#
# 3. K selection rules:
#    → Start with K = √(n_training_samples)
#    → Always pick ODD K for classification (avoids ties)
#    → Use cross-validation to confirm the best K
#
# 4. ALWAYS scale before KNN — largest range feature
#    dominates all distances without scaling
#
# 5. Main weakness: slow at prediction time (O(n) per
#    prediction), keeps entire dataset in RAM always
#
# 6. INTERVIEW TRAP:
#    Q: "Why should K be an odd number in KNN
#        classification?"
#    Right answer: "To avoid TIE situations. If K=4 and
#    2 neighbors vote 'Yes' and 2 vote 'No', the model
#    can't decide. With K=3 or K=5, there's always a
#    clear majority winner. This doesn't matter for
#    regression since we take the average, not a vote."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → KNN Hyperparameters
#    Tuning distance metrics, weights, and algorithm
#    parameters to squeeze the best performance from KNN.
# ============================================================