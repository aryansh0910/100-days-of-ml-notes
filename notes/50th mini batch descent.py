# ============================================================
# MINI-BATCH GRADIENT DESCENT
# ============================================================

# ============================================================
# WHAT IS IT?
#
# ITS LIKE THERE ARE 1000 ROWS AND U MADE A BATCH OF 100,
# SO THERE WOULD BE 10 BATCHES PER EPOCH. SO THERE WOULD BE
# ONE UPDATE ON THE LOSS FUNCTION IN EACH BATCH, SO THERE
# WOULD BE 10 UPDATES IN EACH EPOCH IN THIS CASE. SO IT'S
# LIKE A MIDDLE THING BETWEEN THE BOTH.
#
# Mini-Batch Gradient Descent is the "Balanced Professional."
# It is the industry standard used in almost all deep
# learning and large-scale machine learning today.
# ============================================================


# ============================================================
# 1. THE CORE CONCEPT
#
# Instead of using the whole dataset (Batch) or just one
# row (Stochastic), you split your data into small "snacks"
# called batches. You calculate the error and update your
# weights only after seeing ONE batch.
#
# COMMON BATCH SIZES: 32, 64, 128, or 256 rows.
#
# THE LOGIC: You get enough data to make a stable decision
# (unlike SGD), but you do it fast enough to keep the
# momentum going (unlike Batch GD).
#
# VISUAL — 1000 rows, batch size 100:
#
# Epoch 1:
# [Batch 1: rows 1-100]   → update 1
# [Batch 2: rows 101-200] → update 2
# [Batch 3: rows 201-300] → update 3
#  ...
# [Batch 10: rows 901-1000] → update 10
#
# Total updates in 1 epoch = 10 (not 1 like Batch GD,
# not 1000 like SGD — right in the middle!)
# ============================================================


# ============================================================
# WHY MINI-BATCH WINS — THREE REASONS
#
# 1. HARDWARE EFFICIENCY (GPU POWER):
#    Modern computers use "Vectorization." Your CPU and GPU
#    are designed to do math on matrices of 32 or 64 rows
#    almost as fast as they do math on 1 row. Using SGD
#    (1 row) actually WASTES your computer's power.
#
# 2. STABILITY WITH SPEED:
#    Much less "jumpy" than SGD. The path to the minimum is
#    relatively direct, but still has enough "noise" to
#    jump out of local minima (the "fake valleys").
#
# 3. SMOOTH CONVERGENCE:
#    While SGD bounces around the bottom forever, Mini-Batch
#    (with a decaying learning rate) settles down quite nicely.
#
# VISUAL — hardware utilization:
#
# SGD (1 row at a time):
# GPU: [██░░░░░░░░░░░░░░░░░░] 10% utilized — wasteful
#
# Mini-Batch (64 rows at a time):
# GPU: [██████████████████░░] 90% utilized — efficient
#
# Batch GD (all 1M rows at once):
# GPU: [██████████████████████] crashes — too much for RAM
# ============================================================


# ============================================================
# 3. KEY TERMINOLOGY
#
# BATCH SIZE: The number of training examples used in
#             ONE iteration (e.g. 32)
#
# ITERATION:  ONE single update of the weights (after
#             seeing one batch)
#
# EPOCH:      ONE complete pass through the entire dataset
#
# EXAMPLE: If you have 1,000 rows and a Batch Size of 100,
# one Epoch consists of 10 Iterations.
#
# VISUAL — the hierarchy:
#
# EPOCH (1 full pass through all 1000 rows)
#   └── Iteration 1 (Batch of 100 rows) → 1 weight update
#   └── Iteration 2 (Batch of 100 rows) → 1 weight update
#   └── ...
#   └── Iteration 10 (Batch of 100 rows) → 1 weight update
# ============================================================


# ============================================================
# FULL COMPARISON — ALL THREE TYPES
#
# ┌──────────────┬─────────────────┬────────────────┬────────────────┐
# │ Metric       │ Batch GD        │ Stochastic GD  │ Mini-Batch GD  │
# ├──────────────┼─────────────────┼────────────────┼────────────────┤
# │ Rows/update  │ All rows        │ 1 row          │ 32-256 rows    │
# │ Speed        │ Slowest         │ Fastest        │ Optimal        │
# │ Path to      │ Smooth & Direct │ Chaotic &      │ Mostly Smooth  │
# │ Minima       │                 │ Jumpy          │                │
# │ Hardware     │ Underutilized   │ Underutilized  │ Fully Utilized │
# │              │                 │                │ (Parallelism)  │
# └──────────────┴─────────────────┴────────────────┴────────────────┘
# ============================================================


# ============================================================
# WHICH SKLEARN METHOD MAPS TO WHICH TYPE
#
# THE LOGISTIC REGRESSION USES THE BATCH
# THE SGD .fit() USES THE STOCHASTIC ONE
# THE SGD .partial_fit() USES THE MINI-BATCH ONE
#
# WHY?
# .fit() on SGDRegressor/SGDClassifier processes the data
# one row at a time internally (true SGD behavior)
# .partial_fit() lets YOU control the batch — you call it
# repeatedly, feeding it one mini-batch chunk at a time,
# which mimics true mini-batch behavior
# ============================================================


# ============================================================
# CODE
# ============================================================

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Sample data
np.random.seed(42)
X = np.random.rand(1000, 1) * 10
y = 3 * X.flatten() + 5 + np.random.randn(1000) * 2

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Mini-Batch implementation using partial_fit
sgd = SGDRegressor(max_iter=1, learning_rate='constant',
                   eta0=0.01, random_state=42)

batch_size = 100
n_epochs = 10
n_samples = len(X_scaled)

for epoch in range(n_epochs):
    # Shuffle indices each epoch for better convergence
    indices = np.random.permutation(n_samples)

    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start:start + batch_size]
        X_batch = X_scaled[batch_idx]
        y_batch = y[batch_idx]

        # partial_fit updates weights using just this batch
        sgd.partial_fit(X_batch, y_batch)

print("Final weight:", sgd.coef_)
print("Final bias:", sgd.intercept_)
# 10 epochs × 10 batches = 100 total weight updates
# (compare: Batch GD would do only 10 updates total,
#  pure SGD would do 10,000 updates total)


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Mini-Batch GD splits data into small chunks (32, 64,
#    128, 256 rows) — updates weights once per chunk
#
# 2. Industry standard for deep learning and large-scale ML
#
# 3. Three reasons it wins:
#    → Hardware efficiency (GPU vectorization)
#    → Stable but still has noise to escape local minima
#    → Smoother convergence than pure SGD
#
# 4. Key terms: Batch Size, Iteration, Epoch
#    1000 rows, batch size 100 → 10 iterations per epoch
#
# 5. sklearn mapping:
#    SGDRegressor.fit()         → true Stochastic GD
#    SGDRegressor.partial_fit() → Mini-Batch GD (call
#                                  repeatedly per batch)
#
# 6. INTERVIEW TRAP:
#    Q: "Why not just always use the smallest possible
#        batch size for fastest training?"
#    Wrong: "Smaller batch = always faster"
#    Right: "Too small a batch (like SGD's 1 row) wastes
#    GPU/CPU vectorization power — hardware is optimized to
#    process matrices of 32-256 rows almost as fast as 1
#    row. So very small batches are actually computationally
#    inefficient despite seeming 'lighter.' The sweet spot
#    balances stability, speed, and hardware utilization."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Polynomial Regression
#    Extending Linear Regression to fit curved relationships
#    using polynomial features.
# ============================================================