# ============================================================
# UNDERFITTING AND OVERFITTING
# ============================================================

# ============================================================
# WHAT IS IT?
# Underfitting and Overfitting are the two most common
# problems models face during training.
#
# Think of it as a SPECTRUM:
#
# Too Simple ←————————————————→ Too Complex
# (Underfitting)   Sweet Spot   (Overfitting)
#      ↑                              ↑
#  High Bias                    High Variance
# ============================================================


# ============================================================
# 1. UNDERFITTING — The "Lazy" Student
# ============================================================
#
# DEFINITION:
# The model is too simple to learn the underlying patterns
# in the data. It fails on BOTH training and test data.
#
# ANALOGY:
# A student who only skims the headings of a textbook.
# When the exam comes, they can't answer basic questions
# because they never learned the details.
#
# THE SIGN:
# High Training Error + High Test Error
#
# TECHNICAL TERM: High Bias
# The model makes too many assumptions.
# Example: "I'll assume all cars under 5 years cost 10 Lakhs"
# → Way too simple, misses the real pattern completely.
#
# REAL WORLD EXAMPLE:
# Using a straight line to predict house prices in a city
# where prices depend on 20+ factors → model is too simple
# to capture that complexity.
# ============================================================


# ============================================================
# 2. OVERFITTING — The "Ratta" / Memorizer Student
# ============================================================
#
# DEFINITION:
# The model is too complex and learns the NOISE/RANDOMNESS
# in training data instead of the actual pattern.
# Performs perfectly on training data but fails on new data.
#
# ANALOGY:
# A student who memorizes every question and answer
# word-for-word. If the teacher changes the numbers
# in the exam, the student fails because they didn't
# understand the logic — they just "recorded" the data.
#
# THE SIGN:
# Low Training Error + High Test Error
#
# TECHNICAL TERM: High Variance
# The model is too sensitive to small changes in training data.
#
# REAL WORLD EXAMPLE:
# A model trained on 100 rows that gets 99% training accuracy
# but 60% test accuracy → it memorized those 100 rows
# instead of learning the general pattern.
# ============================================================


# ============================================================
# 3. THE SWEET SPOT — The Ideal Student
# ============================================================
#
# DEFINITION:
# The model captures the GENERAL TREND without getting
# distracted by the noise (random errors in data).
#
# ANALOGY:
# A student who understands the concepts.
# They might get 90% in practice and 88% in the final exam.
# They can handle new, unseen questions because they learned
# the RULES, not the ANSWERS.
#
# THE SIGN:
# Low Training Error + Low Test Error
# (and the GAP between them is small)
#
# THIS IS WHAT WE ARE ALWAYS TRYING TO ACHIEVE.
# ============================================================


# ============================================================
# HOW TO DETECT THEM — The Score Check
# ============================================================

# This is used to check the training score vs the testing score
print("Training Score :", model.score(X_train, y_train))
print("Test Score     :", model.score(X_test, y_test))

# HOW TO READ THE OUTPUT:
#
# Train = 0.99, Test = 0.70  →  OVERFITTING  ❌
# Train = 0.50, Test = 0.50  →  UNDERFITTING ❌
# Train = 0.92, Test = 0.89  →  SWEET SPOT   ✅
#
# RULE OF THUMB:
# If the GAP between train and test score is more than
# 10-15% → you have an overfitting problem.
# ============================================================


# ============================================================
# HOW TO FIX THEM
# ============================================================

# ── IF YOU ARE UNDERFITTING ──────────────────────────────────
#
# 1. Increase Complexity
#    → Use a more powerful model
#    → Example: switch from Linear Regression to Random Forest
#
# 2. Add More Features
#    → Give the model more information
#    → Example: in Car Dekho, add "Owner Type" if not already
#
# 3. Train Longer
#    → Let the model learn for more iterations/epochs
#
# 4. Reduce Regularization
#    → If penalty is too strong, model becomes too simple

# ── IF YOU ARE OVERFITTING ───────────────────────────────────
#
# 1. Simplify the Model
#    → Reduce number of features or depth of trees
#    → Use a simpler algorithm
#
# 2. Get More Data
#    → Harder to memorize 1,000,000 rows than 100 rows
#    → More data = model is forced to learn general patterns
#
# 3. Regularization
#    → Add penalties for being too complex
#    → L1, L2 regularization (like l2_leaf_reg in CatBoost)
#    → Dropout in Neural Networks
#
# 4. Early Stopping
#    → Stop training as soon as Test Error starts going UP
#    → Don't let the model over-learn the training data
#
# 5. Cross Validation
#    → Evaluate on multiple splits, not just one
#    → Gives a more honest picture of model performance
# ============================================================


# ============================================================
# VISUAL MENTAL MODEL
# ============================================================
#
#        Loss
#          |
#  High →  |  \      ← Underfitting zone (both high)
#           |   \
#           |    \  ← Sweet spot (small gap, both low)
#           |     \————
#  Low  →   |          \————  ← Overfitting zone
#           |                  (train low, test goes up)
#           |_________________________________ Complexity
#
# As complexity increases:
# → Underfitting reduces
# → After a point, Overfitting begins
# → Sweet spot is in the middle
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Underfitting = model too simple = High Bias
#    Sign: High Train Error + High Test Error
#
# 2. Overfitting = model too complex = High Variance
#    Sign: Low Train Error + High Test Error
#
# 3. Sweet Spot = what we always aim for
#    Sign: Low Train Error + Low Test Error (small gap)
#
# 4. Always check: model.score(X_train) vs model.score(X_test)
#    Gap > 15% → overfitting problem
#
# 5. Fix Underfitting → more complexity, more features
#    Fix Overfitting  → regularization, more data, early stop
#
# 6. These concepts apply to ALL models:
#    Linear Regression, Random Forest, Neural Networks — ALL
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Cross Validation
#    Instead of one train/test split, we split the data
#    multiple times and average the results for a more
#    honest and reliable evaluation of our model.
# ============================================================