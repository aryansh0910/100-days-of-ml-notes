# ============================================================
# HOW TO CHOOSE THE RIGHT ALGORITHM
# ============================================================

# ============================================================
# WHAT IS IT?
#
# This is the "Million-Dollar Question" in Machine Learning.
# In higher dimensions (e.g. 20+ features), you can't
# physically SEE the data, so you have to use Mathematical
# Proxies and Validation Strategies to "feel" the shape
# of your data.
#
# Think of it as being in a DARK ROOM: you can't see the
# furniture, but you can throw balls (test points) to see
# how they bounce.
# ============================================================


# ============================================================
# 1. THE "LINEARITY CHECK" — SVM vs OTHERS
#
# How do you know if a straight "sheet" (Hyperplane) can
# separate your data?
#
# THE SIMPLE BASELINE TEST:
# Run a Logistic Regression or a Linear SVM.
# → 90% accuracy → data is MOSTLY LINEAR ✅
# → 50% accuracy (random guessing) → data is "Swirled"
#   or "Clumpy" (NON-LINEAR) ❌
#
# THE KERNEL TEST:
# Run SVM with Linear kernel AND SVM with RBF kernel.
# → RBF significantly better than Linear → data has
#   COMPLEX CURVED relationships a flat plane can't catch
#
# VISUAL:
#
# Linear data (line works):    Non-linear data (line fails):
#  ● ● ●  |  ○ ○ ○              ○ ○ ○ ○ ○
#  ● ● ●  |  ○ ○ ○             ○  ● ● ●  ○
#  ● ● ●  |  ○ ○ ○             ○  ● ● ●  ○
#           ↑ line works         ○ ○ ○ ○ ○
#                                 ↑ line fails, need RBF
# ============================================================


# ============================================================
# 2. THE "BRANCHING" CHECK — DECISION TREES / RANDOM FOREST
#
# Decision Trees don't care about "Distance" or "Lines."
# They care about THRESHOLDS (e.g. Is Glucose > 120?).
#
# INDICATOR: Use Trees if your data is TABULAR (like Excel)
# and has CATEGORICAL features (e.g. Gender, City, Blood Type).
#
# THE "INTERACTION" SIGN:
# If the target depends on a COMBINATION of factors
# (e.g. "If Age > 50 AND Smoker = Yes"), Trees/Forests are
# much better than SVM at catching these "branches"
# automatically.
#
# VISUAL — interaction that trees catch, SVM misses:
#
# Rule: "Churn IF (Tenure < 12) AND (Monthly_Bill > 80)"
#
# SVM tries to find ONE boundary for the whole space ❌
# Tree naturally creates:
# [Tenure < 12?]
#      YES → [Monthly_Bill > 80?]
#                  YES → CHURN ✅
#                  NO  → STAY
#      NO  → STAY
# ============================================================


# ============================================================
# 3. THE "CLUMPINESS" CHECK — KNN
#
# KNN assumes that "Like attracts Like." It works best
# when the data is clustered into TIGHT LOCAL GROUPS.
#
# THE "LAZY" CHECK: If your data has a very high "Local
# Correlation" (meaning a patient is almost always like
# their 5 closest neighbors), KNN will SHINE.
#
# THE WARNING: If you have 100+ features, KNN usually
# FAILS (The Curse of Dimensionality) because in high
# dimensions, every point becomes "far" from every other.
#
# VISUAL:
#
# Good for KNN (tight clumps):    Bad for KNN (spread out):
#  ●●●    ○○○                      ● ○ ● ○ ● ○ ●
#  ●●●    ○○○     ✅               ○ ● ○ ● ○ ● ○  ❌
#  ●●●    ○○○                      ● ○ ● ○ ● ○ ●
#  (clear neighborhoods)           (no clear neighborhoods)
# ============================================================


# ============================================================
# DECISION TABLE — WHICH ALGORITHM WHEN
#
# ┌──────────────────────┬───────────────────┬─────────────────────────────┐
# │ If your data is...   │ Best Algorithm    │ Why?                        │
# ├──────────────────────┼───────────────────┼─────────────────────────────┤
# │ Linear & Simple      │ Linear SVM /      │ Fast, stable, less prone    │
# │                      │ Logistic Reg      │ to overfitting              │
# │ Complex & Non-Linear │ SVM (RBF Kernel)  │ Warps space to find         │
# │                      │                   │ boundary                    │
# │ Categorical &        │ Random Forest /   │ Handles "If/Then" logic     │
# │ Branched             │ XGBoost           │ perfectly                   │
# │ Small & Clumpy       │ KNN               │ Uses local similarity,      │
# │                      │                   │ no "training" needed        │
# │ Large dataset        │ Logistic Reg /    │ SVM and KNN are too slow    │
# │ (100k+ rows)         │ Random Forest     │ for large data              │
# │ High dimensional     │ SVM / Random      │ KNN fails due to curse of   │
# │ (100+ features)      │ Forest            │ dimensionality              │
# └──────────────────────┴───────────────────┴─────────────────────────────┘
# ============================================================


# ============================================================
# 5. THE PROFESSIONAL WAY — THE LEARNING CURVE
#
# The MOST RELIABLE way to know is to plot a Learning Curve.
#
# 1. Train ALL models (SVM, RF, KNN, Logistic Reg)
# 2. Compare their Cross-Validation Scores
# 3. The one with the HIGHEST score on the Test/Validation
#    set is the WINNER. Your data "is" whatever that
#    algorithm is best at finding.
# ============================================================


# ============================================================
# CODE — COMPARING ALL ALGORITHMS AT ONCE
# ============================================================

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reg)

models = {
    'Logistic Regression': LogisticRegression(),
    'SVM Linear'         : SVC(kernel='linear'),
    'SVM RBF'            : SVC(kernel='rbf'),
    'Random Forest'      : RandomForestClassifier(n_estimators=100),
    'KNN'                : KNeighborsClassifier(n_neighbors=5)
}

print("=" * 50)
print("ALGORITHM COMPARISON — Cross Validation R2")
print("=" * 50)

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y_reg,
                             cv=5, scoring='r2')
    print(f"{name:25s} → Mean: {np.mean(scores):.4f}"
          f"  Std: {np.std(scores):.4f}")

# The algorithm with the HIGHEST mean CV score wins
# High std → unstable model, be cautious


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. LINEARITY CHECK:
#    Linear model gives ~90% → data is linear → use
#    Logistic Reg or Linear SVM
#    Linear gives ~50% → non-linear → try RBF kernel
#
# 2. If RBF >> Linear in accuracy → complex curved data
#    → SVM RBF is the right choice
#
# 3. BRANCHING CHECK: tabular data with categorical
#    features or interaction effects → Random Forest /
#    XGBoost always beats SVM and KNN here
#
# 4. CLUMPINESS CHECK: tight local clusters → KNN shines
#    but FAILS with 100+ features (curse of dimensionality)
#
# 5. PROFESSIONAL WAY: don't guess — run all models,
#    compare cross-validation scores, let the data decide
#
# 6. INTERVIEW TRAP:
#    Q: "How do you choose between SVM, KNN, and Random
#        Forest for a new dataset?"
#    Right answer: "I never guess — I run a baseline
#    comparison using cross-validation on all three. But
#    I also use these heuristics: if a linear model already
#    performs well, SVM or Logistic Regression. If the data
#    is tabular with categorical features or interaction
#    effects, Random Forest. If the dataset is small and
#    has tight local clusters, KNN. If 100+ features are
#    present, I rule out KNN immediately due to the curse
#    of dimensionality. The CV scores confirm the final
#    choice."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Naive Bayes
#    A probabilistic classifier based on Bayes' theorem —
#    extremely fast and the go-to for text classification.
# ============================================================