# ============================================================
# WHY DOES A MODEL UNDERFIT AND OVERFIT?
# ============================================================

# ============================================================
# WHAT IS IT?
# A common confusion for beginners:
# "How can a model be both too lazy (underfitting)
#  AND too obsessed (overfitting)?"
#
# The answer lies in HYPERPARAMETERS and TRAINING TIME.
# Think of the model like a "brain" that you can tune
# to be either very simple or very complex.
# You are in CONTROL of where it lands on the spectrum.
# ============================================================


# ============================================================
# 1. HOW UNDERFITTING HAPPENS — The Simple Brain
# ============================================================
#
# DEFINITION:
# Underfitting occurs when you RESTRICT the model so much
# that it can't see the complex patterns in your data.
# It's like trying to draw a circle with a ruler —
# the tool is too "stiff."
#
# CAUSES:
#
# → Low Complexity:
#   You set the parameters to be too small.
#   Example: in CatBoost, depth=1 or iterations=10
#   The model only learns one or two basic rules
#   and ignores everything else.
#
# → Too Much Regularization:
#   You added too much "penalty" (like very high l2_leaf_reg)
#   This makes the model so scared of being wrong
#   that it REFUSES to learn any detailed patterns.
#   (Like a student too scared to attempt any question)
#
# → Short Training:
#   You stopped training after 5 seconds.
#   The model didn't have enough "time" to look
#   at the data properly.
#
# RESULT:
# High Train Error + High Test Error
# ============================================================


# ============================================================
# 2. HOW OVERFITTING HAPPENS — The Obsessive Brain
# ============================================================
#
# DEFINITION:
# Overfitting occurs when you give the model TOO MUCH FREEDOM.
# It starts to think that random "noise" or typos in your
# data are actually important rules.
#
# CAUSES:
#
# → Excessive Complexity:
#   You set the parameters too high.
#   Example: in CatBoost, depth=15
#   The model now has the "brain power" to memorize every
#   single car's unique license plate or owner name
#   instead of learning general rules about age and engine.
#
# → Training for Too Long:
#   If you set iterations=100,000, the model will eventually
#   run out of real patterns to learn and will start
#   "memorizing" the exact position of every dot on your graph.
#
# → Small Dataset:
#   If you only have 100 rows but a very complex model,
#   the model can easily find a "fake" pattern that only
#   exists in those 100 rows but nowhere else in the world.
#
# RESULT:
# Low Train Error + High Test Error
# ============================================================


# ============================================================
# 3. THE SIGNAL vs NOISE CONCEPT
# ============================================================
#
# To understand WHY this happens, you must realize that
# ALL data has two parts:
#
# SIGNAL → The REAL patterns
#   Example: "Older cars are cheaper"
#   This is the truth that exists in the real world.
#   This is what we WANT the model to learn.
#
# NOISE  → Random luck / errors / outliers
#   Example: "This one specific old car was expensive
#   because the owner was a celebrity"
#   This is a random exception, NOT a real pattern.
#   This does NOT exist in the real world consistently.
#
# ┌─────────────────────────────────────────────┐
# │ Underfitting → FAILS to find the Signal     │
# │ Overfitting  → MISTAKES Noise for Signal    │
# │ Sweet Spot   → Finds Signal, ignores Noise  │
# └─────────────────────────────────────────────┘
#
# ANALOGY:
# Signal = the actual melody of a song
# Noise  = background static and distortion
#
# Underfitting = model can't even hear the melody
# Overfitting  = model memorizes the static too
# Sweet Spot   = model hears the melody, ignores static
# ============================================================


# ============================================================
# 4. HOW TO FIND THE SWEET SPOT — Validation Curves
# ============================================================
#
# This is where VALIDATION CURVES come in.
# You plot the error of training data AND test data
# on the same graph as you increase complexity:
#
#        Error
#          |
#  High →  |\                          /
#           | \         Test Error    /
#           |  \                     /
#           |   \__________________/
#           |    \                      ← Test error rising
#           |     \____                    = Overfitting zone
#  Low  →   |          ————————————   ← Train error keeps dropping
#           |___________________________________ Complexity →
#                  ↑           ↑
#            Underfitting   GOLDILOCKS
#              zone           ZONE ✅
#
# HOW TO READ IT:
# Both errors HIGH         → Underfitting zone ❌
# Train error = 0,
# Test error HIGH          → Overfitting zone  ❌
# Both errors LOW,
# small gap between them   → Goldilocks Zone   ✅
#
# The point RIGHT BEFORE the test error starts rising
# again is your GOLDILOCKS ZONE — this is where you
# want your model to be!
# ============================================================


# ============================================================
# PUTTING IT ALL TOGETHER — The Control Panel
# ============================================================
#
# Think of these as knobs you can turn:
#
# KNOB             TURN DOWN →           TURN UP →
# ──────────────────────────────────────────────────
# depth            Underfitting          Overfitting
# iterations       Underfitting          Overfitting
# l2_leaf_reg      Overfitting           Underfitting
# dataset size     Overfitting risk ↑    Overfitting risk ↓
# training time    Underfitting          Overfitting
#
# Your job as an ML engineer = tune these knobs
# until you hit the Goldilocks Zone!
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. The SAME model can underfit OR overfit depending
#    on how you set its hyperparameters and training time.
#    YOU are in control.
#
# 2. Underfitting = model too restricted = Simple Brain
#    Caused by: low depth, too much regularization,
#    short training, too few features
#
# 3. Overfitting = model too free = Obsessive Brain
#    Caused by: high depth, long training,
#    small dataset, too many features
#
# 4. All data = Signal + Noise
#    Underfitting → misses the Signal
#    Overfitting  → memorizes the Noise
#    Sweet Spot   → captures Signal, ignores Noise
#
# 5. Use Validation Curves to VISUALLY find the sweet spot
#    Plot train error vs test error vs complexity
#    Sweet spot = right before test error starts rising
#
# 6. The Goldilocks Zone = not too simple, not too complex
#    This is ALWAYS the goal.
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Cross Validation
#    Instead of one train/test split, we split the data
#    multiple times and average the results for a more
#    honest and reliable evaluation of our model.
# ============================================================