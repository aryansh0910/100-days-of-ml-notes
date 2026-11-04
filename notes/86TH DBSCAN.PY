# ============================================================
# DBSCAN — DENSITY-BASED SPATIAL CLUSTERING
# ============================================================

# ============================================================
# WHAT IS IT?
#
# ITS A UNSUPERVISED CLUSTERING ALGORITHM THAT GROUPS DATA
# POINTS BASED ON DENSITY. IT'S GREAT FOR DISCOVERING
# CLUSTERS OF ARBITRARY SHAPES AND IDENTIFYING OUTLIERS.
# ITS OFTEN BETTER THAN K-MEANS FOR REAL-WORLD DATA
# BECAUSE IT DOESN'T ASSUME CLUSTERS ARE SPHERICAL OR
# EVEN SIMILAR IN SIZE.
#
# WHILE KMEANS DOES THE CLUSTERING BASED ON DISTANCE TO
# CENTROIDS, DBSCAN CLUSTERS BASED ON THE DENSITY OF
# POINTS IN THE SPACE. IT LOOKS FOR "CROWDS" OF POINTS
# AND LABELS THEM AS CLUSTERS, WHILE ISOLATED POINTS ARE
# LABELED AS NOISE (OUTLIERS).
#
# Imagine you have a map of a city with thousands of dots
# representing people, but no labels for neighborhoods.
# DBSCAN is the tool that looks at where the dots are
# "CROWDED" together to find the neighborhoods
# automatically.
# ============================================================


# ============================================================
# THE TWO KEY PARAMETERS
#
# EPSILON: The maximum distance between two points to be
# considered neighbors. Like saying "If you are within
# 5 feet of me, you are in my circle."
# EPSILON IS LIKE THE RADIUS UP TO WHICH IT WILL CHECK
# THE DENSITY OF THE POINTS.
#
# MIN_SAMPLES: The minimum number of points needed to
# form a "dense region." If you have 5 people standing
# together it's a "crowd" (cluster). If it's just 2
# people, it might just be a fluke.
# ============================================================


# ============================================================
# 2. THE THREE TYPES OF POINTS
#
# CORE POINT IS THE POINT IN THE RADIUS (EPSILON) WHICH
# HAS MIN_SAMPLES NUMBER OF POINTS IN IT. LIKE IF THE
# EPSILON IS 0.5 AND MIN_SAMPLES IS 5, THEN A CORE POINT
# IS A POINT THAT HAS AT LEAST 5 OTHER POINTS WITHIN A
# RADIUS OF 0.5.
#
# BORDER POINT IS A POINT THAT HAS LESS THAN MIN_SAMPLES
# NUMBER OF POINTS IN ITS EPSILON RADIUS, BUT HAS AT
# LEAST ONE CORE POINT IN ITS EPSILON RADIUS.
#
# NOISE POINT IS A POINT THAT HAS LESS THAN MIN_SAMPLES
# NUMBER OF POINTS IN ITS EPSILON RADIUS AND DOESN'T HAVE
# ANY CORE POINT IN ITS EPSILON RADIUS. LABELED AS -1.
#
# VISUAL — point types with eps=0.5, min_samples=3:
#
#         ●──●                ×  ← Noise (-1)
#        /    \               (alone, no core nearby)
#       ●  C   ●   B●
#        \    /     |
#         ●──●      C●──●
#                      |
#                      ●
#
# C = Core Point (has ≥ min_samples neighbors in radius)
# B = Border Point (near a core but not enough neighbors)
# × = Noise (isolated, labeled -1)
#
# VISUAL — eps effect:
#
# eps too SMALL:               eps too LARGE:
# × × × × × × ×               [all one giant cluster]
# (everything = noise) ❌      (no separation) ❌
#
# eps just RIGHT:
# [Cluster 1] [Cluster 2] × ×
# (clear groups + noise) ✅
# ============================================================


# ============================================================
# DBSCAN vs K-MEANS
#
# ┌──────────────────┬──────────────────┬───────────────────┐
# │ Feature          │ K-Means          │ DBSCAN            │
# ├──────────────────┼──────────────────┼───────────────────┤
# │ Shape            │ Only finds       │ Finds ANY shape   │
# │                  │ circles/spheres  │ (crescents, lines)│
# │ Outliers         │ Forces outliers  │ Ignores outliers  │
# │                  │ into a group ❌  │ automatically ✅  │
# │ Number of        │ Must tell it K   │ Discovers K for   │
# │ clusters         │ beforehand       │ you automatically │
# │ Speed            │ Very fast        │ Slower on large   │
# │                  │                  │ datasets          │
# └──────────────────┴──────────────────┴───────────────────┘
#
# ITS GENERALLY USED FOR:
# → Anomaly detection
# → Customer segmentation
# → Any situation where you want to discover groups
#   without predefined labels
# ============================================================


# ============================================================
# 5. THE GOLDEN RULE FOR EPS
#
# eps too SMALL → everything becomes Noise (-1)
# eps too LARGE → every point merged into one giant cluster
#
# Finding the perfect eps is the HARDEST part of DBSCAN.
# We use a "K-DISTANCE PLOT" to find the "elbow" where
# the distance suddenly jumps — that's your ideal eps.
# ============================================================


# ============================================================
# CODE — DBSCAN
# ============================================================

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Scaling is MANDATORY for DBSCAN
# (uses distances — same reason as KNN and SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Initialize DBSCAN
# eps         → the 'search radius'
# min_samples → minimum points to form a group
db = DBSCAN(eps=0.5, min_samples=5)

# 3. Fit and Predict
# Clusters labeled 0, 1, 2... and -1 for Noise
clusters = db.fit_predict(X_scaled)

# 4. Check results
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise    = list(clusters).count(-1)

print("Number of clusters found:", n_clusters)
print("Number of noise points (Outliers):", n_noise)
# -1 in clusters → subtract 1 because -1 is noise, not a cluster


# ============================================================
# FINDING THE PERFECT EPS — K-DISTANCE PLOT
#
# FINDING THE PERFECT eps (EPSILON) VALUE IS THE ONLY WAY
# TO MAKE DBSCAN WORK. IF YOU GUESS, YOU'LL EITHER END UP
# WITH ONE GIANT CLUSTER OR 100% NOISE.
#
# The standard trick is the K-DISTANCE GRAPH. We calculate
# the distance to the "kth" nearest neighbor for every
# point and look for the "ELBOW" — the point where the
# distance suddenly spikes.
#
# HOW TO READ THE "ELBOW":
# → The line stays flat for a while (close points)
# → Suddenly curves UPWARD sharply
# → The Y-axis value at that "bend" = your IDEAL EPSILON
#
# If you pick ABOVE the elbow → too forgiving → clusters merge
# If you pick BELOW the elbow → too strict → too much noise
# ============================================================

from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# 1. Set k = min_samples
# Industry standard: min_samples = 2 × number of features
neighbors     = NearestNeighbors(n_neighbors=4)
neighbors_fit = neighbors.fit(X_scaled)

# 2. Find the distances
distances, indices = neighbors_fit.kneighbors(X_scaled)

# 3. Sort the distances (last column = kth distance)
distances = np.sort(distances[:, 3], axis=0)

# 4. Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title("K-Distance Graph to find EPS")
plt.xlabel("Data Points sorted by distance")
plt.ylabel("Epsilon Value (Distance)")
plt.grid(True)
plt.show()
# Look for the ELBOW in this plot
# The Y value at the elbow = your best eps


# ============================================================
# MIN_SAMPLES RULE OF THUMB
#
# STANDARD RULE: MinSamples = 2 × NumberOfFeatures
# MESSY/NOISY DATA: Increase (e.g. 10 or 20) to force
#                   model to only find VERY dense crowds
# SMALL DATASETS: Keep it low (e.g. 3)
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. DBSCAN = density-based, NOT distance-to-centroid based
#    Groups "crowds" of points, labels isolated points as
#    noise (-1)
#
# 2. Three point types:
#    Core Point  → ≥ min_samples within eps radius
#    Border Point → near a core but not enough own neighbors
#    Noise        → alone, labeled -1
#
# 3. Two parameters:
#    eps → search radius (find via K-Distance elbow plot)
#    min_samples → 2 × n_features (rule of thumb)
#
# 4. ALWAYS scale before DBSCAN — distance based
#
# 5. Discovers K automatically — no need to specify
#    number of clusters beforehand unlike K-Means
#
# 6. INTERVIEW TRAP:
#    Q: "When would you use DBSCAN over K-Means?"
#    Right answer: "Three situations. First, when clusters
#    have arbitrary shapes like crescents or rings —
#    K-Means only finds spherical clusters. Second, when
#    you don't know K beforehand — DBSCAN discovers the
#    number of clusters automatically. Third, when you
#    have outliers — K-Means forces every point into a
#    cluster even if it's clearly noise, while DBSCAN
#    labels them -1 so you can identify and remove them.
#    The tradeoff is that DBSCAN requires careful tuning
#    of eps via the K-Distance elbow plot."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Model Selection & Deployment
#    Saving your best model and deploying it as a web app
#    using Streamlit for your Resume Screener project.
# ============================================================