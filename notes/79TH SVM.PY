# ============================================================
# SUPPORT VECTOR MACHINE (SVM)
# ============================================================

# ============================================================
# WHAT IS IT?
#
# SVM STANDS FOR SUPPORT VECTOR MACHINE, AND IT'S A
# POWERFUL ALGORITHM FOR BOTH CLASSIFICATION AND REGRESSION.
#
# ITS LIKE LOGISTIC REGRESSION ON STEROIDS. WHILE LOGISTIC
# REGRESSION TRIES TO FIND A STRAIGHT LINE TO SEPARATE
# CLASSES, SVM TRIES TO FIND THE WIDEST POSSIBLE STREET
# (HYPERPLANE) THAT SEPARATES THEM.
#
# SUPPORT VECTORS ARE THE DATA POINTS THAT ARE CLOSEST TO
# THE DECISION BOUNDARY ON EITHER SIDE. THEY ARE THE MOST
# DIFFICULT TO CLASSIFY AND HAVE THE MOST INFLUENCE ON
# WHERE THE HYPERPLANE IS PLACED.
# ============================================================


# ============================================================
# THE CORE DIAGRAM
#
# GROUP RED (+)         GROUP BLUE (-)
#       * * .    .
#          * (S+) <---------- (S-)    .
#              |       |        |
#     MARGIN   |   DECISION     |   MARGIN
#     BORD (+) |   LINE (0)     |   BORD (-)
#       |      |       |        |      |
#       |      |       |        |      |
#       V      |       V        |      V
#              |                |
#    ----------|-------|--------|----------
#              |   ^   |    ^   |
#              |   |   |    |   |
#              |<--+-->|<---+-->|
#                  |        |
#             HALF-MARGIN  HALF-MARGIN
#
# S+ and S- are the SUPPORT VECTORS — the only points
# that actually determine where the boundary goes.
# SVM aims to MAXIMIZE the total margin width.
#
# THEY HANDLE OUTLIERS BETTER THAN LOGISTIC REGRESSION
# BECAUSE THEY FOCUS ON SUPPORT VECTORS RATHER THAN ALL
# DATA POINTS. THIS MAKES THEM MORE ROBUST TO NOISE.
# ============================================================


# ============================================================
# 1. THE CORE CONCEPT — THE OPTIMAL HYPERPLANE
#
# Imagine you have two groups of points (Healthy vs.
# Diabetic) on a 2D plot. You could draw MANY lines to
# separate them. SVM doesn't just want any line — it wants
# the one that stays as FAR AWAY from both groups as
# possible.
#
# HYPERPLANE: The "decision boundary" (a line in 2D,
# a plane in 3D, a hyperplane in higher dimensions)
#
# SUPPORT VECTORS: Data points right on the EDGE of the
# groups — the most difficult to classify, and the ONLY
# points that determine where the boundary goes
#
# MARGIN: Distance between hyperplane and support vectors.
# SVM MAXIMIZES this margin.
#
# VISUAL — why maximum margin matters:
#
# Line A (any line, small margin):
#  ● ●|○ ○     ← boundary close to points
#   ●●|○ ○       new points near boundary → wrong prediction
#
# Line B (SVM, maximum margin):
#  ● ●    |    ○ ○   ← maximum gap between groups
#   ●●    |    ○ ○     new points have more room → correct
#
# Line B generalizes MUCH better to new data ✅
# ============================================================


# ============================================================
# 2. LINEAR vs NON-LINEAR — THE KERNEL TRICK
#
# What if the data isn't separable by a straight line?
# What if "Healthy" points are in a CIRCLE surrounded by
# "Diabetic" points?
#
# SVM uses a KERNEL to project the data into a HIGHER
# DIMENSION where it becomes linearly separable.
#
# LINEAR KERNEL: A simple straight line.
#
# RBF (Radial Basis Function) Kernel: The most popular.
# Creates circular or wavy boundaries by treating the data
# as if it exists in INFINITE dimensions.
#
# POLYNOMIAL KERNEL: Creates curved boundaries.
#
# VISUAL — the Kernel Trick:
#
# 2D (not separable by line):     3D (separable by plane!):
#   ○ ○ ○ ○                          ○ ○ ○ ○  ← lifted up
#  ○ ● ● ● ○                        ─────────  ← hyperplane
#  ○ ● ● ● ○                         ● ● ● ●  ← stays down
#   ○ ○ ○ ○
#
# Kernel "lifts" the inner points into a third dimension
# where a flat plane can now separate them ✅
# ============================================================


# ============================================================
# 3. THE C HYPERPARAMETER — THE PENALTY
#
# This is the MOST IMPORTANT knob in SVM. It controls the
# tradeoff between "Clean Boundary" and "Margin Width."
#
# LOW C (Soft Margin): Model is "RELAXED." Allows some
# points to be on the wrong side of the boundary if it
# means the overall margin can be WIDER. Better for noisy
# data. → HIGHER BIAS risk
#
# HIGH C (Hard Margin): Model is "STRICT." Tries to
# classify every single point correctly, even if it makes
# the margin very NARROW. → OVERFITTING risk
#
# VISUAL:
#
# Low C (wide margin, some mistakes allowed):
#  ● ●  X          ← X is on wrong side (allowed)
#  ● ●  |  ○ ○     ← wide margin ✅ better generalization
#       |  ○ ○
#
# High C (narrow margin, zero mistakes):
#  ● ●●|○ ○        ← boundary squeezed to fit everything
#   ●●●|○ ○        ← very narrow margin ❌ overfitting risk
#
# SCALING IS VERY MUCH NECESSARY FOR SVM BECAUSE IT RELIES
# ON DISTANCE CALCULATIONS TO FIND THE OPTIMAL HYPERPLANE.
# IF ONE FEATURE HAS A MUCH LARGER SCALE THAN OTHERS, IT
# CAN DOMINATE THE DISTANCE AND LEAD TO POOR PERFORMANCE.
# ============================================================


# ============================================================
# SVR SPECIFIC — EPSILON (THE TUBE WIDTH)
#
# EPSILON defines the width of the "TUBE" (margin of error)
# where NO PENALTY is given.
#
# SMALL EPSILON: Very thin tube. Model tries to get very
# close to every single point. → More sensitive
#
# LARGE EPSILON: Thick tube. Model ignores points that are
# "close enough" to the line, focusing only on big outliers.
#
# VISUAL:
#
# Small epsilon (thin tube):
#        ●   ←── penalty (outside tube)
#   ─────────────  ← upper tube boundary
#   ─────────────  ← regression line
#   ─────────────  ← lower tube boundary
#      ●  ←── no penalty (inside tube)
#
# Large epsilon (thick tube):
#   ─────────────────────────  ← upper tube boundary
#        ●  ←── no penalty (inside thick tube)
#   ─────────────  ← regression line
#      ●  ←── no penalty
#   ─────────────────────────  ← lower tube boundary
# ============================================================


# ============================================================
# PROS AND CONS
#
# ┌──────────────────────┬──────────────────────────────────┐
# │ PROS                 │ CONS                             │
# ├──────────────────────┼──────────────────────────────────┤
# │ Effective in high    │ Not for large datasets: very     │
# │ dimensions: great    │ slow if you have over 50,000     │
# │ when many features   │ rows                             │
# │                      │                                  │
# │ Memory efficient:    │ Sensitive to scale: fails        │
# │ only stores the      │ miserably if data isn't          │
# │ Support Vectors      │ standardized                     │
# │                      │                                  │
# │ Kernel Flexibility:  │ Hard to interpret: unlike a      │
# │ can fit almost any   │ tree, you can't easily "see"     │
# │ shape of data        │ why it made a choice             │
# └──────────────────────┴──────────────────────────────────┘
# ============================================================


# ============================================================
# CODE
# ============================================================

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Scaling is MANDATORY for SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reg)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42
)

# 3. Initialize SVR
# kernel='rbf' → handles non-linear patterns
# C=100        → strict model (low tolerance for mistakes)
# epsilon=0.1  → defines "no-penalty" tube around prediction
svm_reg = SVR(kernel='rbf', C=100, epsilon=0.1)

# 4. Train
svm_reg.fit(X_train, y_train)

# 5. Evaluate
print("SVM Training Score:", svm_reg.score(X_train, y_train))
print("SVM Test Score:", svm_reg.score(X_test, y_test))
# If Train >> Test → C too high, reduce it
# If both low → try RBF kernel or increase C slightly


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. SVM finds the MAXIMUM MARGIN hyperplane — not just
#    any boundary, but the one farthest from both classes
#
# 2. Support Vectors = only the edge points that define
#    the boundary. All other points don't matter!
#
# 3. Kernel Trick → projects data to higher dimensions
#    where linear separation becomes possible
#    RBF is the default and handles most non-linear cases
#
# 4. C controls bias-variance tradeoff:
#    Low C  → wide margin, some mistakes allowed (robust)
#    High C → narrow margin, zero mistakes (overfitting)
#
# 5. Epsilon (SVR only) → tube width where no penalty
#    Small = sensitive, Large = ignores close points
#
# 6. ALWAYS scale before SVM — distance-based, unscaled
#    features completely break it
#
# 7. Not suitable for large datasets (>50k rows) — very
#    slow since it computes distances between all points
#
# 8. INTERVIEW TRAP:
#    Q: "What are Support Vectors and why do they matter?"
#    Right answer: "Support Vectors are the data points
#    closest to the decision boundary on each side. They
#    are the ONLY points that determine where the hyperplane
#    goes — if you removed all other points, the boundary
#    would stay exactly the same. This is also why SVM is
#    memory efficient — it only needs to store these few
#    critical points, not the entire dataset like KNN."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → SVM Hyperparameters
#    Tuning C, kernel, epsilon and gamma to get the best
#    performance from your SVM model.
# ============================================================