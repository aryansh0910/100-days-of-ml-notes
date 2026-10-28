# ============================================================
# K-MEANS CLUSTERING + KNN
# ============================================================

# ============================================================
# K-MEANS CLUSTERING
#
# CLUSTERING ALGORITHMS IS AN UNSUPERVISED LEARNING
# TECHNIQUE, WHERE THE GOAL IS TO GROUP SIMILAR DATA POINTS
# TOGETHER BASED ON THEIR FEATURES.
#
# STEPS:
# 1. DECIDE THE NUMBER OF CLUSTERS (K)
# 2. RANDOMLY INITIALIZE K CENTROIDS (CENTERS OF CLUSTERS)
# 3. ASSIGN EACH DATA POINT TO THE NEAREST CENTROID
#    (USING DISTANCE CALCULATIONS)
# 4. UPDATE THE CENTROIDS BY CALCULATING THE MEAN OF ALL
#    DATA POINTS ASSIGNED TO EACH CLUSTER
# 5. REPEAT STEPS 3 AND 4 UNTIL THE CENTROIDS DO NOT
#    CHANGE SIGNIFICANTLY OR A MAXIMUM NUMBER OF
#    ITERATIONS IS REACHED
#
# VISUAL — how K-Means works step by step:
#
# Step 1-2: Random centroids placed
# ┌─────────────────────────────┐
# │  ●  ●    ✦        ●  ●     │  ✦ = centroid
# │    ●        ✦  ●    ●      │
# │  ●   ●             ●       │
# └─────────────────────────────┘
#
# Step 3: Assign each point to nearest centroid
# ┌─────────────────────────────┐
# │  A  A    ✦A       B  B     │
# │    A        ✦B  B    B     │
# │  A   A             B       │
# └─────────────────────────────┘
#
# Step 4: Move centroids to mean of their cluster
# ┌─────────────────────────────┐
# │  A  A    ✦A       B  B     │  ← ✦ moved to true center
# │    A     ✦B    B    B      │
# │  A   A             B       │
# └─────────────────────────────┘
# Repeat until centroids stop moving
# ============================================================


# ============================================================
# HOW TO DECIDE K — THE ELBOW METHOD
#
# NOW HOW TO DECIDE THE NUMBER OF CLUSTERS (K):
# WE USE THE ELBOW METHOD, WHERE WE PLOT THE
# WITHIN-CLUSTER-SUM-OF-SQUARES (ALSO KNOWN AS INERTIA)
# (WCSS) AGAINST DIFFERENT VALUES OF K.
# THE POINT WHERE THE WCSS STARTS TO DECREASE MORE SLOWLY
# (LIKE AN ELBOW) IS USUALLY A GOOD CHOICE FOR K.
#
# INERTIA MEASURES HOW WELL THE DATA POINTS ARE CLUSTERED
# AROUND THE CENTROIDS. LOWER INERTIA MEANS BETTER
# CLUSTERING, BUT IT CAN ALSO LEAD TO OVERFITTING IF K
# IS TOO HIGH.
#
# WHEN AFTER INCREASING THE NUMBER OF CLUSTERS THE INERTIA
# DECREASES SIGNIFICANTLY THEN IT IS A GOOD CHOICE FOR K,
# BUT IF THE INERTIA DECREASES SLOWLY THEN IT IS NOT A
# GOOD CHOICE FOR K. SO WE CHOOSE THE ELBOW POINT WHERE
# THE INERTIA DECREASES SIGNIFICANTLY, BUT AFTER THAT IT
# STARTS TO DECREASE SLOWLY.
#
# VISUAL — Elbow curve:
#
# Inertia
#   |●
#   | ●
#   |  ●
#   |   ●
#   |    ●●
#   |      ●●●●●●●●  ← elbow here, pick this K
#   |_________________________ K
#   1  2  3  4  5  6  7  8
#          ↑
#     best K (elbow point)
# ============================================================


# ============================================================
# CODE — K-MEANS + ELBOW METHOD
# ============================================================

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Scale data (Distances matter here just as much as KNN!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reg)

# 2. List to store the 'Inertia' values
inertia_values = []

# 3. Test different numbers of clusters (1 to 10)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)
    # .inertia_ = sum of squared distances to centroid

# 4. Plot the Elbow
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia_values, marker='o', color='purple')
plt.title('The Elbow Method (K-Means Inertia)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()


# ============================================================
# KNN — K-NEAREST NEIGHBORS
# ============================================================

# ============================================================
# WHAT IS IT?
#
# KNN is one of the most "human-like" algorithms. While
# other models try to find a mathematical formula or a tree
# structure, KNN uses very simple logic:
# "Tell me who your NEIGHBORS are, and I'll tell you who
# YOU are."
# ============================================================


# ============================================================
# 1. THE CORE PHILOSOPHY
#
# KNN is a LAZY LEARNER. It doesn't actually "learn"
# anything during training. It just MEMORIZES the entire
# dataset. When you ask it to make a prediction:
#
# 1. Looks at the entire "map" of your data
# 2. Finds the K CLOSEST points (neighbors)
# 3. Takes the AVERAGE (Regression) or MAJORITY VOTE
#    (Classification) of those neighbors
#
# VISUAL:
#
# New point ★ needs a prediction (K=3):
#
#  ●(Churn)  ●(Stay)
#       ●(Churn)  ★  ←── find 3 nearest neighbors
#  ●(Stay)       ●(Churn)
#
# 3 nearest: Churn, Churn, Stay
# Majority vote → CHURN ✅
# ============================================================


# ============================================================
# 2. THE "DISTANCE" MATH
#
# How does it know who is "close"? It uses distance
# formulas. The most common is EUCLIDEAN DISTANCE
# (the straight-line distance).
#
# For two points (x1, y1) and (x2, y2):
# Distance = √((x2-x1)² + (y2-y1)²)
#
# Because it relies on distance, SCALING YOUR DATA IS
# MANDATORY. If one column is "Age" (0 to 100) and another
# is "Annual Income" (0 to 100,000), the Income column will
# COMPLETELY DOMINATE the distance math.
# ============================================================


# ============================================================
# 4. CHOOSING THE RIGHT K
#
# The value of K determines the "smoothness" of your model:
#
# SMALL K (e.g. K=1): Very sensitive. Follows every tiny
# outlier. → HIGH VARIANCE (Overfitting)
#
# LARGE K (e.g. K=100): Too "blurry." Just averages almost
# everyone. → HIGH BIAS (Underfitting)
#
# VISUAL:
#
# K=1 (overfitting):           K=100 (underfitting):
# Decision boundary is         Decision boundary is
# jagged, follows noise        too smooth, misses details
#  ●  ●                         ●  ●
#   ○●○  ●                       ○○○  ●
#  ○  ○●                        ○  ○○
# (wiggly, noisy)               (too smooth, inaccurate)
#
# Sweet spot: usually K = √n (square root of dataset size)
# or use cross-validation to find the best K
# ============================================================


# ============================================================
# PROS AND CONS
#
# ┌─────────────────────┬──────────────────────────────────┐
# │ PROS                │ CONS                             │
# ├─────────────────────┼──────────────────────────────────┤
# │ Simple: extremely   │ SLOW: as dataset grows, must     │
# │ easy to understand  │ calculate distance to every point│
# │                     │                                  │
# │ No Assumptions:     │ MEMORY HOG: needs whole dataset  │
# │ works for linear    │ in RAM at all times              │
# │ and non-linear data │                                  │
# │                     │ SENSITIVE TO NOISE: one bad      │
# │ Versatile: works    │ outlier can mess up the local    │
# │ for both regression │ neighborhood                     │
# │ and classification  │                                  │
# └─────────────────────┴──────────────────────────────────┘
# ============================================================


# ============================================================
# 6. THE CURSE OF DIMENSIONALITY
#
# KNN works great with 2, 5, or 10 features. But as you
# add more columns (like 100+), the "distance" between
# points starts to become MEANINGLESS. In high-dimensional
# space, everyone ends up being "far" from everyone else.
#
# This is why we often use PCA or Feature Selection BEFORE
# running KNN.
#
# VISUAL:
#
# 2D (KNN works great):
# ★ is close to ● and far from ○ → clear neighbors
#
# 100D (KNN breaks down):
# Distance to nearest neighbor  ≈ 48 units
# Distance to farthest neighbor ≈ 52 units
# → barely any difference → KNN can't tell who's "close"
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# K-MEANS:
# 1. Unsupervised — no labels needed, just finds groups
# 2. Steps: init centroids → assign → update → repeat
# 3. Use Elbow Method to find best K — pick where inertia
#    stops dropping sharply
# 4. ALWAYS scale before K-Means — uses distances
#
# KNN:
# 1. Lazy learner — memorizes data, no training phase
# 2. Prediction = average (regression) or vote (classification)
#    of K nearest neighbors
# 3. ALWAYS scale — dominated by largest range column
# 4. Small K → overfitting, Large K → underfitting
# 5. Fails in high dimensions — use PCA first
#
# 6. INTERVIEW TRAP:
#    Q: "What is the difference between K-Means and KNN?"
#    Right answer: "Completely different algorithms despite
#    both using K. K-Means is UNSUPERVISED clustering —
#    it groups unlabeled data into K clusters by moving
#    centroids. KNN is SUPERVISED — it uses labeled training
#    data to predict the class/value of new points by
#    finding the K nearest neighbors. K-Means learns
#    cluster structure, KNN memorizes the entire dataset
#    and uses distances at prediction time."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → KNN Hyperparameters
#    Tuning K, distance metrics, and weights to get the
#    best performance from your KNN model.
# ============================================================