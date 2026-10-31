# ============================================================
# HIERARCHICAL CLUSTERING (AGGLOMERATIVE)
# ============================================================

# ============================================================
# WHAT IS IT?
#
# In the world of Clustering, if K-Means is like "sorting
# things into K buckets," Agglomerative Hierarchical
# Clustering is like building a FAMILY TREE.
#
# It is a BOTTOM-UP approach. It starts by treating every
# single data point as its own individual cluster and then
# slowly MERGES them together based on similarity until
# everyone belongs to one big group.
# ============================================================


# ============================================================
# 1. HOW THE ALGORITHM WORKS — STEP BY STEP
#
# STEP 1 — Individual Clusters:
# Every data point starts as a cluster of 1.
#
# STEP 2 — Find the Pair:
# The algorithm finds the two "CLOSEST" clusters.
#
# STEP 3 — Merge:
# It merges those two into a single, larger cluster.
#
# STEP 4 — Repeat:
# Repeats until only ONE cluster (containing all points)
# remains.
#
# VISUAL — bottom-up merging:
#
# Start: Each point is its own cluster
# ●  ●  ●  ●  ●  ●
# A  B  C  D  E  F
#
# Step 1: A and B are closest → merge
# [AB]  ●  ●  ●  ●
#       C  D  E  F
#
# Step 2: E and F are closest → merge
# [AB]  ●  ●  [EF]
#       C  D
#
# Step 3: C and D merge
# [AB]  [CD]  [EF]
#
# Step 4: [AB] and [CD] merge
# [ABCD]  [EF]
#
# Step 5: Final merge
# [ABCDEF]  ← one big cluster
#
# The DENDROGRAM records every one of these merges!
# ============================================================


# ============================================================
# 2. THE DENDROGRAM — THE "FAMILY TREE"
#
# This is the MOST IMPORTANT part of Hierarchical
# Clustering. A Dendrogram is a diagram that records every
# merge the algorithm made.
#
# HORIZONTAL LINES: Represent a MERGE between two clusters
# VERTICAL LINES: The HEIGHT represents the DISTANCE
# between the clusters when they were merged
#
# THE RULE: The TALLER the vertical line, the more
# "DIFFERENT" the two groups were when merged.
#
# VISUAL — reading a Dendrogram:
#
# Distance
#   |                    ┌──────────┐
# 8 |                    │          │
#   |           ┌────────┘          │
# 5 |           │                   │
#   |     ┌─────┘        ┌──────────┘
# 3 |     │              │
#   |  ┌──┘        ┌─────┘
# 1 |  │           │
#   |  A  B        C  D  E  F
#
# HOW TO READ IT:
# A and B merged at distance 1 → very similar ✅
# C,D,E,F merged at distance 3 → somewhat similar
# Two big groups merged at distance 8 → quite different
#
# HOW TO CHOOSE K:
# Find the LONGEST vertical line that no horizontal line
# crosses → draw a horizontal cut there → number of lines
# you cut = number of clusters
#
# In the visual above, cutting at distance 6 gives 2
# clusters: [AB] and [CDEF]
# ============================================================


# ============================================================
# 3. LINKAGE — HOW TO MEASURE "DISTANCE" BETWEEN GROUPS
#
# Since you are merging GROUPS of points, how do you
# measure the distance between two groups?
#
# WARD'S LINKAGE (Default):
# Minimizes the VARIANCE within the clusters. Usually
# creates the most even, compact groups. BEST DEFAULT.
#
# SINGLE LINKAGE:
# Measures the distance between the TWO CLOSEST points
# in each cluster. Can create long, "stringy" clusters.
#
# COMPLETE LINKAGE:
# Measures the distance between the TWO FARTHEST points.
# Creates more compact clusters than Single.
#
# AVERAGE LINKAGE:
# Measures the AVERAGE distance between ALL points in
# both clusters. A compromise between Single and Complete.
#
# VISUAL — same two clusters, different linkage results:
#
# Cluster A: ●●●          Cluster B: ●●●
#               ●                   ●
#
# Single   → measures shortest gap (A's edge to B's edge)
# Complete → measures longest gap (A's far side to B's far side)
# Average  → measures average of all pairwise distances
# Ward     → measures how much merging would increase variance
# ============================================================


# ============================================================
# K-MEANS vs HIERARCHICAL — WHEN TO USE WHICH
#
# ┌──────────────────┬──────────────────┬───────────────────┐
# │                  │ K-Means          │ Hierarchical      │
# ├──────────────────┼──────────────────┼───────────────────┤
# │ K selection      │ Must decide K    │ Decide K AFTER    │
# │                  │ BEFORE running   │ seeing dendrogram │
# │ Shape of         │ Circular/convex  │ Any shape         │
# │ clusters         │ clusters only    │                   │
# │ Speed            │ Fast, scales     │ Slow on large     │
# │                  │ to large data    │ datasets (O(n³))  │
# │ Interpretability │ Less (just       │ High — dendrogram │
# │                  │ labels)          │ shows full history│
# │ Best for         │ Large datasets,  │ Small datasets,   │
# │                  │ known K          │ exploring structure│
# └──────────────────┴──────────────────┴───────────────────┘
# ============================================================


# ============================================================
# CODE
# ============================================================

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# 1. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reg)

# 2. Generate the Linkage Matrix (The 'History' of merges)
# Ward linkage is usually the most robust
Z = linkage(X_scaled, method='ward')
# Z records every merge: which clusters merged, at what
# distance, and how many points are in the new cluster

# 3. Plot the Dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Dendrogram for Patient Clusters')
plt.xlabel('Patient Index')
plt.ylabel('Distance (Ward)')
plt.show()
# Look for the LONGEST vertical line with no horizontal
# crossing it → that's where to cut → that many clusters

# 4. Apply the Clustering
# Based on the Dendrogram, we decide to cut into 3 clusters
cluster_model = AgglomerativeClustering(n_clusters=3,
                                         linkage='ward')
labels = cluster_model.fit_predict(X_scaled)

print("First 10 Cluster Labels:", labels[:10])
# Each data point gets a label 0, 1, or 2 (its cluster)


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Hierarchical Clustering = bottom-up, starts with N
#    clusters (one per point), merges until 1 remains
#
# 2. Dendrogram records every merge:
#    → Taller vertical line = more different the groups were
#    → Cut the longest uninterrupted vertical line
#      to decide number of clusters
#
# 3. Linkage types:
#    Ward (default) → minimizes variance, most compact
#    Single → closest points, can be stringy
#    Complete → farthest points, more compact than single
#    Average → compromise between single and complete
#
# 4. Big advantage over K-Means: you DON'T need to decide
#    K before running — the dendrogram shows you the
#    natural structure of the data first
#
# 5. Big disadvantage: slow on large datasets (O(n³))
#    K-Means is much faster for large data
#
# 6. Always scale before clustering — uses distances
#
# 7. INTERVIEW TRAP:
#    Q: "What is the advantage of Hierarchical Clustering
#        over K-Means?"
#    Right answer: "Two main advantages. First, you don't
#    need to specify K beforehand — you run the algorithm,
#    visualize the dendrogram, and THEN decide K by finding
#    the longest vertical line. Second, the dendrogram gives
#    you the full merging history, so you can understand the
#    natural grouping structure of your data at multiple
#    levels — K-Means just gives you flat labels with no
#    insight into the hierarchy."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → DBSCAN
#    A density-based clustering algorithm that can find
#    clusters of any shape and automatically identifies
#    outliers as noise points.
# ============================================================