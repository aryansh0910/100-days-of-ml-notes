# ============================================================
# SVM — THE KERNEL TRICK
# ============================================================

# ============================================================
# WHAT IS IT?
#
# The Kernel Trick is the "magic" that allows SVM to solve
# complex, non-linear problems without actually doing the
# impossible math of higher dimensions. It's the reason
# SVM became a legend in machine learning.
# ============================================================


# ============================================================
# THE PROBLEM — "MARBLE CAKE" DATA
#
# Imagine you have a plate of red and blue marbles. The red
# ones are in the CENTER, and the blue ones form a RING
# around them. No matter how you draw a straight line on
# that plate, you can NEVER separate the red from the blue.
#
# VISUAL — non-separable in 2D:
#
#     ○ ○ ○ ○ ○
#    ○  ● ● ●  ○
#    ○  ● ● ●  ○      ← no straight line can separate
#    ○  ● ● ●  ○        ● from ○
#     ○ ○ ○ ○ ○
# ============================================================


# ============================================================
# 2. THE SOLUTION — ADD A DIMENSION
#
# What if you could "LIFT" the red marbles into the air?
# Suddenly, you could slide a flat sheet of paper (a
# Hyperplane) UNDERNEATH the red marbles but ABOVE the
# blue ones.
#
# In 2D → data was INSEPARABLE
# In 3D → a flat plane separates them PERFECTLY
#
# THE RBF FUNCTION IN THE SVM KERNEL TRICK CREATES THIS
# "LIFTING" EFFECT BY CALCULATING DISTANCES IN A HIGHER-
# DIMENSIONAL SPACE WITHOUT ACTUALLY MOVING THE DATA THERE.
# IT'S LIKE USING A MAGIC LENS TO SEE THE SEPARATION
# WITHOUT CHANGING THE REALITY OF THE DATA.
#
# U CAN ALSO USE POLYNOMIAL KERNELS TO CREATE CURVED
# BOUNDARIES, OR SIGMOID KERNELS TO MIMIC NEURAL NETWORKS.
# THE CHOICE OF KERNEL DEPENDS ON THE SHAPE OF YOUR DATA
# AND THE COMPLEXITY OF THE RELATIONSHIPS YOU WANT TO
# CAPTURE.
#
# VISUAL — lifting into 3D:
#
# 2D (inseparable):         After kernel (3D view):
#  ○ ○ ○ ○ ○               ● ● ● ●   ← lifted UP
#  ○  ●●●  ○         →     ─────────  ← hyperplane
#  ○  ●●●  ○               ○ ○ ○ ○   ← stays DOWN
#  ○ ○ ○ ○ ○
# ============================================================


# ============================================================
# 3. WHAT IS THE "TRICK"?
#
# Here is the GENIUS part: Projecting data into 3D (or
# 1000D) is mathematically expensive and SLOW. The Kernel
# Trick allows us to calculate the "relationship" (the dot
# product) between points in that high-dimensional space
# WITHOUT EVER ACTUALLY MOVING THE DATA THERE.
#
# It uses a mathematical shortcut (a Kernel Function) to
# "CHEAT" and get the 3D result while staying in 2D.
#
# ANALOGY: Instead of physically building a 3D model of
# a city to measure distances, you use a mathematical
# formula that gives you the same distance answer while
# looking at a flat map. Same result, no 3D construction!
# ============================================================


# ============================================================
# KERNEL TYPES
#
# ┌──────────────┬───────────────────────────┬──────────────┐
# │ Kernel Type  │ What it does              │ Best for...  │
# ├──────────────┼───────────────────────────┼──────────────┤
# │ Linear       │ Draws a straight          │ Simple,      │
# │              │ line/plane                │ linear data  │
# │ Polynomial   │ Draws curved boundaries   │ Data with    │
# │              │ (parabolas)               │ curved       │
# │              │                           │ relationships│
# │ RBF (Radial) │ Creates "bubbles" or      │ Most complex │
# │              │ "islands" around points   │ non-linear   │
# │              │                           │ data (Default│
# │ Sigmoid      │ Acts like a Neural        │ Specific     │
# │              │ Network layer             │ proxy cases  │
# └──────────────┴───────────────────────────┴──────────────┘
# ============================================================


# ============================================================
# 6. THE GAMMA PARAMETER (SPECIFIC TO RBF)
#
# When using the Kernel Trick, Gamma is your FOCUS:
#
# HIGH GAMMA: The "bubble" around each point is very SMALL.
# The model only cares about IMMEDIATE neighbors.
# → Risk of OVERFITTING (too sensitive to local points)
#
# LOW GAMMA: The "bubble" is LARGE. The model considers
# points far away, creating a smoother, simpler boundary.
# → Risk of UNDERFITTING
#
# VISUAL — Gamma effect:
#
# High Gamma (small bubbles, jagged boundary):
#  ●  ●   ●     ○   ○
#  (○)  (●)  (○)        ← each point has tiny influence
#   ↑ jagged, wiggly boundary — overfits ❌
#
# Low Gamma (large bubbles, smooth boundary):
#  ● ● ●  |  ○ ○ ○      ← smooth straight-ish boundary
#          ↑ generalizes well ✅
#
# RELATIONSHIP WITH C:
# High C + High Gamma → very likely to overfit
# Low C  + Low Gamma  → very likely to underfit
# Tune BOTH together using GridSearchCV
# ============================================================


# ============================================================
# CODE — RBF KERNEL
# ============================================================

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 1. Scale data (MANDATORY for kernels to work)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reg)

# 2. Use the RBF Kernel (The most common "Trick")
# gamma='scale' → automatically sets gamma based on
#                  number of features and variance
# C=1.0         → moderate penalty (balanced)
svm_kernel = SVC(kernel='rbf', C=1.0, gamma='scale')

# 3. Train
svm_kernel.fit(X_scaled, y_reg)
print("Model trained using the RBF Kernel Trick.")


# ============================================================
# CODE — GRIDSEARCHCV WITH MULTIPLE KERNELS
#
# CHECKING MULTIPLE PARAMETERS WITH DIFFERENT SPECIFIC
# PARAMETERS IN EACH WITH A LIST OF DICTIONARIES
# ============================================================

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# Define SEPARATE grids for each kernel
# Why list of dicts? Because each kernel has DIFFERENT
# valid parameters — poly needs 'degree', rbf needs 'gamma'
# If you put them all in one dict, sklearn tries invalid
# combinations like linear + gamma → wastes time
param_grid = [
    {
        'kernel': ['linear'],
        'C'     : [0.1, 1, 10]
        # No gamma or degree needed for linear
    },
    {
        'kernel': ['rbf'],
        'C'     : [0.1, 1, 10],
        'gamma' : [0.01, 0.1, 1]
        # No degree needed for rbf
    },
    {
        'kernel': ['poly'],
        'C'     : [0.1, 1, 10],
        'degree': [2, 3, 4],    # degree is ONLY for poly
        'coef0' : [0, 1, 2]
        # coef0 = independent term in polynomial kernel
    }
]

grid = GridSearchCV(SVR(), param_grid, cv=5,
                    scoring='neg_mean_squared_error')
# grid.fit(X_train, y_train)
# grid.best_params_ → tells you best kernel + params combo


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Kernel Trick = mathematical shortcut that computes
#    distances in high-dimensional space WITHOUT physically
#    projecting data there — same result, much faster
#
# 2. RBF kernel = default and most powerful — creates
#    "bubbles" around points, handles any non-linear shape
#
# 3. Gamma (RBF specific):
#    High gamma → small bubbles → overfitting
#    Low gamma  → large bubbles → underfitting
#    Tune C and gamma TOGETHER
#
# 4. Use LIST OF DICTS in GridSearchCV when different
#    kernels have different valid parameters — prevents
#    sklearn from testing invalid combinations
#
# 5. Kernel choice:
#    Linear → when data is already separable
#    RBF    → default, handles most cases
#    Poly   → when you know relationships are polynomial
#    Sigmoid → rarely used in practice
#
# 6. INTERVIEW TRAP:
#    Q: "What is the Kernel Trick and why is it needed?"
#    Right answer: "Some data is not linearly separable in
#    its current dimensions — like points in a ring. The
#    Kernel Trick implicitly maps data to a higher
#    dimension where it BECOMES linearly separable. The
#    'trick' is that it computes the DOT PRODUCT in that
#    high-dimensional space using a simple kernel function,
#    WITHOUT ever actually projecting the data there. This
#    avoids the massive computational cost of working in
#    thousands of dimensions directly."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Naive Bayes
#    A probabilistic classifier based on Bayes' theorem,
#    extremely fast and great for text classification.
# ============================================================