# ============================================================
# CURSE OF DIMENSIONALITY
# ============================================================

# ============================================================
# WHAT IS IT?
#
# The Curse of Dimensionality refers to the various phenomena
# that arise when analyzing and organizing data in
# high-dimensional spaces (lots of columns/features) that
# do not occur in low-dimensional settings.
#
# IT MEANS IF U INCREASE THE NUMBER OF FEATURES, UR RESULT
# MIGHT DECREASE.
#
# FEATURES LIKE TEXT DATA AND IMAGE DATA HAVE A LOT OF
# COLUMNS, LIKE A LOT OF ZEROS.
# ============================================================


# ============================================================
# THE INTUITION — "THE EMPTY SPACE PROBLEM"
#
# Imagine you have 100 pieces of candy:
#
# 1D (A Line):
#   Spread them on a 1-meter line — very close together.
#   Finding a pattern is easy.
#
# 2D (A Square):
#   Spread them on a 1m x 1m table. They start to look
#   sparse.
#
# 3D (A Cube):
#   Put them in a 1m x 1m x 1m box. Most of the box is now
#   empty air.
#
# 100D:
#   The space becomes so massive that your 100 data points
#   are like lonely stars in different galaxies.
#
# THE RESULT: Your model can't find patterns because every
# data point looks "far away" from every other data point.
#
# VISUAL:
#
# 1D:  ●●●●●●●●●●●●●●●●●●●●  ← dense, easy to find pattern
#
# 3D:    ●        ●
#           ●  ●        ●
#      ●        ●     ●     ← already sparse
#
# 100D: ●                          ●
#                  ●
#            ●              ●     ← essentially empty space,
#                                    every point isolated
# ============================================================


# ============================================================
# HOW IT AFFECTS YOUR CHURN MODEL
#
# You used OneHotEncoder on the City column.
#
# If you have 3 cities, you add 2 columns. No big deal.
# If you had 500 cities, you would add 499 columns.
# Suddenly, your dataset goes from 10 features to 510.
# This triggers the curse:
#
# 1. OVERFITTING:
#    With so many dimensions, a model (like Random Forest)
#    can find a "path" to perfectly explain the training
#    data by basically memorizing it — but it will fail
#    miserably on the test data.
#
# 2. DISTANCE DILUTION:
#    Algorithms that rely on distance (KNN, SVM) break down.
#    In high dimensions, the distance between the NEAREST
#    neighbor and the FARTHEST neighbor becomes almost the
#    same. Everything is "equally far."
#
# 3. COMPUTATIONAL EXPLOSION:
#    The time it takes to run GridSearchCV will grow
#    exponentially.
#
# VISUAL — Distance Dilution:
#
# Low dimensions (2D):
# Nearest neighbor  = 2 units away
# Farthest neighbor = 50 units away
# → CLEAR difference, KNN can pick meaningful neighbors ✅
#
# High dimensions (500D):
# Nearest neighbor  = 48 units away
# Farthest neighbor = 52 units away
# → barely any difference! KNN can't tell who's "close" ❌
# ============================================================


# ============================================================
# HOW DO WE FIGHT THE CURSE?
#
# Since we can't just delete data blindly, we use
# Dimensionality Reduction:
# ============================================================


# ============================================================
# A. FEATURE SELECTION — "SURVIVAL OF THE FITTEST"
#
# You drop columns that don't help.
#
# CORRELATION CHECK:
# If two columns are 95% identical, delete one.
# (They're carrying the same information twice — redundant.)
#
# FEATURE IMPORTANCE:
# Use your Random Forest to tell you which columns it
# actually uses, and drop the "zeros."
#
# Example:
# Random Forest feature_importances_:
# Age        → 0.35  ← keep, model uses this heavily
# Salary     → 0.28  ← keep
# City_xyz   → 0.00  ← drop, model never uses this
# City_abc   → 0.00  ← drop
# ============================================================

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Example workflow
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# importance_df = pd.DataFrame({
#     'feature': X_train.columns,
#     'importance': rf.feature_importances_
# }).sort_values('importance', ascending=False)
# print(importance_df)
# Drop columns where importance is near 0


# ============================================================
# B. FEATURE EXTRACTION — "THE SUMMARY METHOD"
#
# You combine many columns into a few "Super Columns."
#
# PCA (Principal Component Analysis):
# Squashes 100 columns into 5 "components" that capture
# the most variance.
#
# LDA (Linear Discriminant Analysis):
# Similar to PCA but focuses on keeping the classes
# (Churn vs No Churn) SEPARABLE — uses the target label
# to decide which combination of features matters most.
#
# DIFFERENCE PCA vs LDA:
# PCA → unsupervised, just maximizes variance
# LDA → supervised, maximizes class separability
# ============================================================

from sklearn.decomposition import PCA

# Example workflow
# pca = PCA(n_components=5)
# X_reduced = pca.fit_transform(X_train)
# 100 columns → 5 components, most info preserved


# ============================================================
# FEATURE SELECTION vs FEATURE EXTRACTION
#
# ┌─────────────────┬──────────────────┬───────────────────┐
# │                 │ Selection        │ Extraction        │
# ├─────────────────┼──────────────────┼───────────────────┤
# │ What it does    │ Drops columns    │ Combines columns   │
# │ Original columns│ Kept (subset)    │ Lost (transformed) │
# │ Interpretability│ High (same names)│ Low (PC1, PC2...)  │
# │ Examples        │ Correlation,     │ PCA, LDA           │
# │                 │ Feature Importance│                   │
# └─────────────────┴──────────────────┴───────────────────┘
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Curse of Dimensionality = more features can DECREASE
#    model performance, not always increase it
#
# 2. Two big symptoms:
#    → Overfitting (model memorizes instead of learning)
#    → Distance dilution (KNN/SVM lose meaning of "close")
#
# 3. OneHotEncoder on high-cardinality columns is a common
#    trigger — 500 cities = 499 new columns!
#
# 4. Two fixes:
#    Feature Selection  → drop useless columns (keeps names)
#    Feature Extraction → combine columns (PCA, LDA)
#
# 5. PCA is unsupervised (ignores target)
#    LDA is supervised (uses target to maximize separability)
#
# 6. INTERVIEW TRAP:
#    Q: "Does adding more features always improve a model?"
#    Wrong: "Yes, more features = more information"
#    Right: "No — beyond a point, more features cause the
#    curse of dimensionality: overfitting, distance dilution
#    in distance-based algorithms, and exploding compute time.
#    Use feature selection or PCA to manage dimensionality."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → PCA in depth
#    The math behind variance maximization, eigenvectors,
#    and choosing the right number of components.
# ============================================================