# ============================================================
# BATCH GRADIENT DESCENT vs STOCHASTIC GRADIENT DESCENT
# ============================================================

# ============================================================
# BATCH GRADIENT DESCENT — WHAT IS IT?
#
# BATCH GRADIENT DESCENT ==> TYPE OF GRADIENT DESCENT
# (TYPES ARE BASED ON THE SELECTION OF ROWS FOR CALCULATING
# THE SLOPE IN THE FORMULA ==> mx + b)
#
# BATCH GRADIENT CALCULATES THE ERROR ON EACH ROW TO UPDATE
# THE WEIGHTS ONLY ONE SINGLE TIME I.E. UPDATING THE SLOPE
# ONCE PER FULL PASS.
#
# Batch Gradient Descent is the "perfectionist" version of
# optimization. While other versions of Gradient Descent
# are frantic and jumpy, Batch GD is calm, calculated,
# and slow.
# ============================================================


# ============================================================
# 1. THE CORE CONCEPT — BATCH GD
#
# 1. Start with random weights
# 2. Sum up the errors for ALL m data points
# 3. Update the weights ONCE
# 4. Repeat for the next "Epoch" (a full pass through the data)
#
# In Batch Gradient Descent, the algorithm looks at EVERY
# SINGLE ROW in your training dataset before making a
# single update to the weights (w) and bias (b).
#
# If you have 10,000 customers in your churn dataset, Batch
# GD calculates the error for all 10,000, averages them,
# and then takes ONE step down the mountain.
#
# FEATURE TABLE:
#
# ┌─────────────┬──────────────────────────────────────────┐
# │ Feature     │ Performance                               │
# ├─────────────┼──────────────────────────────────────────┤
# │ Stability   │ High. Path to minimum is smooth & direct │
# │ Convergence │ Guaranteed to reach global minimum for   │
# │             │ convex functions (like Linear Regression)│
# │ Memory      │ Heavy. Entire dataset must be loaded     │
# │             │ into RAM to calculate the sum — crashes  │
# │             │ on "Big Data."                            │
# │ Speed       │ Slow. 1 million rows = 1 million          │
# │             │ calculations for just ONE step.           │
# └─────────────┴──────────────────────────────────────────┘
#
# VISUAL — Cost Function as a 3D bowl:
#
# Batch GD looks like a marble being dropped — it rolls
# in a perfectly straight, smooth line toward the lowest
# point.
#
#   ●
#    ╲
#     ╲
#      ╲___________ minimum (smooth, direct path)
# ============================================================


# ============================================================
# STOCHASTIC GRADIENT DESCENT (SGD)
#
# Stochastic Gradient Descent (SGD) is a "speed freak."
# It's chaotic, noisy, and fast.
#
# ITS USED MORE BECAUSE IT IS NOT CALCULATION HEAVY AND
# TIME CONSUMING LIKE BATCH GRADIENT DESCENT.
#
# IT DOESN'T GIVE US STEADY SOLUTIONS, AS THE RANDOM ROW
# IT SELECTS MAY CHANGE EVERY TIME FOR CALCULATING THE ERROR.
# ============================================================


# ============================================================
# 1. THE CORE CONCEPT — SGD
#
# In SGD, instead of waiting to see every single row of
# data before taking a step, the algorithm picks ONE
# random row, calculates the error for just that row, and
# updates the weights IMMEDIATELY.
#
# Batch GD: "I will read the entire book before I form
#            an opinion."
# Stochastic GD: "I will read one sentence and change my
#                 mind instantly."
#
# Because you are updating based on only one data point,
# the "slope" you calculate is a very rough estimate.
# Sometimes that one point is an outlier, and it actually
# sends the model in the wrong direction for a second.
#
# FEATURE TABLE:
#
# ┌──────────────┬─────────────────────────────────────────┐
# │ Feature      │ Performance                              │
# ├──────────────┼─────────────────────────────────────────┤
# │ Speed        │ Insanely fast. Thousands of steps in    │
# │              │ the time Batch GD takes ONE.            │
# │ Memory       │ Very efficient. Only one row in RAM at  │
# │              │ a time. Perfect for "Big Data."          │
# │ Stability    │ Low. Error doesn't go down smoothly;    │
# │              │ it bounces around like a pinball.       │
# │ Global Minima│ Great. The "noise" actually helps the   │
# │              │ model jump out of "Local Minima" (fake  │
# │              │ valleys) to find the real "Global       │
# │              │ Minimum."                                │
# └──────────────┴─────────────────────────────────────────┘
# ============================================================


# ============================================================
# WHY IS SGD USED MORE — THE EPOCH MATH
#
# SO THE NO. OF ITERATIONS REQUIRED (EPOCHS) IS MUCH LESS
# THAN IN BATCH ONE, AS IN EACH EPOCH THE LOSS FUNCTION
# CHANGES N TIMES FOR N ROWS (IN EACH EPOCH), MAKING IT
# CLOSER TO THE CORRECT ONE WITHOUT NEEDING MANY EPOCHS.
#
# IF U COMPARE BOTH WITH THE FIXED NO. OF EPOCHS, LET'S
# SAY M TIMES, AND THERE ARE N ROWS IN THE DATASET:
#
# Batch GD       → changes the loss function only M times
#                  (once per epoch)
# Stochastic GD  → changes it M×N times (once per ROW,
#                  every epoch)
#
# BUT IN REALITY, SGD REQUIRES A LOT LESS EPOCHS COMPARED
# TO THE BATCH VERSION, SO IT'S A LOT FASTER OVERALL.
#
# VISUAL — updates per epoch:
#
# Batch GD,    1 epoch, N=10000 rows:
# █ (1 update — averaged over all 10000 rows)
#
# SGD,         1 epoch, N=10000 rows:
# █████████████████████████████ (10000 updates — one per row)
# ============================================================


# ============================================================
# USES OF SGD
#
# → IN CASE OF BIG DATA WE GENERALLY USE STOCHASTIC
#   GRADIENT DESCENT
#
# → ITS USED IN ALGOs WHERE THE FUNCTION IS NOT CONVEX,
#   THAT IS IT HAS MORE THAN ONE MINIMA KNOWN AS LOCAL
#   MINIMA AND GLOBAL MINIMA. AS IT JUMPS AND ZIG-ZAGS, IT
#   HAS A HIGHER CHANCE TO FIND THE GLOBAL MINIMA FOR THE
#   BEST SOLUTION, WHEREAS BATCH VERSION IS VERY SMOOTH SO
#   IT MAY GET STUCK IN THE LOCAL MINIMA.
#
# SGDRegressor IS AN EXAMPLE OF THIS.
#
# VISUAL — escaping local minima:
#
#  Error
#    |  ╲                    ╱
#    |    ╲     local       ╱
#    |      ╲  minima      ╱
#    |        ╲___      __╱
#    |             \  /
#    |              \/      ← global minimum (deeper, real best)
#
# Batch GD (smooth path) → might settle in local minima,
#                          never escapes because no randomness
#
# SGD (noisy path) → the random jumps can ACCIDENTALLY
#                    bounce it out of a local minima and
#                    into the deeper global minimum ✅
# ============================================================


# ============================================================
# 4. GEOMETRIC VIEW — "THE DRUNKARD'S WALK"
#
# If you look at the path SGD takes on a contour map:
# → It doesn't walk in a straight line
# → It zig-zags and wanders
# → It never truly "settles" at the bottom; it keeps
#   dancing around the minimum point
#
# Stochastic GD looks like a drunk person stumbling toward
# the bottom, zig-zagging everywhere.
#
# VISUAL:
#
# Batch GD:     ●──────────→●  (straight, direct)
#
# SGD:          ●─╮
#                  ╰─╮  ╭─╮
#                     ╰─╯  ╰─╮
#                            ╰──→●  (zigzag, never fully settles)
# ============================================================


# ============================================================
# 5. WHY IS THIS USEFUL FOR YOUR CHURN PROJECT?
#
# If your dataset grows to millions of rows, Batch GD will
# freeze your computer. SGD allows you to train the model
# ONLINE (learning from new customers one by one as they
# sign up).
#
# IMPORTANT NOTE: Because SGD is so jumpy, you often need
# to use a LEARNING RATE SCHEDULER. You start with a big
# step to move fast, and as you get closer to the bottom,
# you make the steps smaller and smaller so the model stops
# bouncing and settles down.
# ============================================================


# ============================================================
# CODE
# ============================================================

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample data
np.random.seed(42)
X = np.random.rand(1000, 1) * 10
y = 3 * X.flatten() + 5 + np.random.randn(1000) * 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Always scale before SGD — it's gradient based!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# SGDRegressor — sklearn's implementation of SGD
sgd = SGDRegressor(max_iter=1000, learning_rate='constant',
                   eta0=0.01, random_state=42)
sgd.fit(X_train_scaled, y_train.ravel())

print("Weight:", sgd.coef_)
print("Bias:", sgd.intercept_)


# ============================================================
# BATCH vs SGD — QUICK COMPARISON
#
# ┌─────────────────┬──────────────────┬───────────────────┐
# │                 │ Batch GD         │ Stochastic GD     │
# ├─────────────────┼──────────────────┼───────────────────┤
# │ Rows per update │ ALL rows         │ ONE random row    │
# │ Path to minimum │ Smooth, direct   │ Noisy, zigzag     │
# │ Speed per epoch │ Slow             │ Fast              │
# │ Memory usage    │ Heavy (full data)│ Light (one row)   │
# │ Local minima    │ Can get stuck    │ Escapes easily    │
# │ Best for        │ Small datasets   │ Big data, online  │
# │                 │                  │ learning          │
# └─────────────────┴──────────────────┴───────────────────┘
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Batch GD: uses ALL rows per update → smooth path,
#    guaranteed convergence for convex functions, but slow
#    and memory heavy
#
# 2. SGD: uses ONE random row per update → fast, memory
#    efficient, but noisy and never fully settles
#
# 3. With M epochs and N rows:
#    Batch GD → M total updates
#    SGD      → M×N total updates (but needs far fewer
#               epochs in practice, making it faster overall)
#
# 4. SGD's randomness HELPS escape local minima — the noise
#    is actually a feature, not a bug, for non-convex
#    functions
#
# 5. SGD needs a Learning Rate Scheduler — big steps early,
#    smaller steps as it approaches the minimum, to stop
#    the bouncing and let it settle
#
# 6. Always scale your features before using SGD — it's
#    gradient based and sensitive to feature scale
#
# 7. INTERVIEW TRAP:
#    Q: "Is SGD always better than Batch GD because it's
#        faster?"
#    Wrong: "Yes, faster is always better"
#    Right: "Not necessarily. Batch GD gives a smooth,
#    guaranteed path to the global minimum for convex
#    problems. SGD trades stability for speed and memory
#    efficiency — great for big data and escaping local
#    minima in non-convex problems, but the noisy
#    convergence means it never perfectly settles without
#    a learning rate scheduler."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Mini-Batch Gradient Descent
#    Combining the stability of Batch GD with the speed
#    of SGD by using small batches of data per update.
# ============================================================