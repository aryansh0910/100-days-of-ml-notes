# ============================================================
# NAIVE BAYES
# ============================================================

# ============================================================
# WHAT IS IT?
#
# Moving from the world of "geometry and margins" (SVM)
# into the world of "PROBABILITY AND COUNTING."
#
# Naive Bayes is the favorite of the data science world
# for one reason: it is LIGHTNING FAST. While an SVM might
# take minutes to train on a massive dataset, Naive Bayes
# finishes in MILLISECONDS because it's essentially just
# doing high-speed multiplication.
# ============================================================


# ============================================================
# 1. THE CORE LOGIC — BAYES' THEOREM
#
# Naive Bayes is based on Bayes' Theorem, a mathematical
# formula used to determine the probability of a
# "Hypothesis" (e.g. this patient has diabetes) given
# "Evidence" (e.g. their glucose is 150).
#
# It calculates:
# → The probability of "Diabetes" happening in general
# → The probability of having "Glucose 150" IF you have
#   Diabetes
# → The probability of having "Glucose 150" in general
#
# It then combines these to give the final answer:
# "There is an 85% chance this patient has diabetes."
#
# VISUAL — the calculation flow:
#
# Prior:    P(Diabetes) = 0.30
#           (30% of people in dataset have diabetes)
#
# Likelihood: P(Glucose=150 | Diabetes) = 0.70
#           (70% of diabetic patients have glucose=150)
#
# Evidence: P(Glucose=150) = 0.25
#           (25% of ALL patients have glucose=150)
#
# Posterior: P(Diabetes | Glucose=150)
#          = (0.70 × 0.30) / 0.25
#          = 0.84 → 84% chance of Diabetes ✅
#
# The model does this for EVERY class and picks the
# one with the HIGHEST posterior probability.
# ============================================================


# ============================================================
# 2. WHY IS IT "NAIVE"?
#
# The "Naive" part comes from a MASSIVE ASSUMPTION:
# It assumes every feature is COMPLETELY INDEPENDENT of
# every other feature.
#
# IN REALITY: If a patient is Overweight (Feature 1),
# they likely have High Blood Pressure (Feature 2).
# These features are "LINKED."
#
# THE NAIVE ASSUMPTION: The model treats Weight and Blood
# Pressure as if they have ABSOLUTELY ZERO CONNECTION.
#
# PEER NOTE: Even though this assumption is technically
# "WRONG," the model is SHOCKINGLY ACCURATE, especially
# for Text Classification and Spam Filtering.
#
# VISUAL — why independence assumption works for spam:
#
# Email contains: "FREE" + "WIN" + "PRIZE"
#
# Naive Bayes:
# P(Spam | FREE) × P(Spam | WIN) × P(Spam | PRIZE)
# = 0.9 × 0.85 × 0.95 = 0.726 → SPAM ✅
#
# Even if "FREE" and "WIN" are correlated in spam emails,
# multiplying their individual probabilities still pushes
# the spam probability high enough to classify correctly.
# This is why "naive" still works in practice!
# ============================================================


# ============================================================
# 3. THE THREE FLAVORS OF NAIVE BAYES
#
# Depending on what your data looks like, you pick a
# different "flavor" of the algorithm:
#
# ┌────────────────┬─────────────────────┬─────────────────────┐
# │ Model Type     │ Best For...         │ Example             │
# ├────────────────┼─────────────────────┼─────────────────────┤
# │ Gaussian NB    │ Bell-curve data     │ Predicting health   │
# │                │ (Numbers/Decimals)  │ from BP or BMI      │
# │ Multinomial NB │ Word counts or      │ Counting how many   │
# │                │ frequency           │ times "Free" or     │
# │                │                     │ "Win" appears       │
# │ Bernoulli NB   │ Binary (Yes/No)     │ Does the email have │
# │                │ data                │ an attachment?      │
# │                │                     │ (1 or 0)            │
# └────────────────┴─────────────────────┴─────────────────────┘
#
# WHEN TO USE WHICH:
# Continuous features (height, weight, glucose) → Gaussian
# Text classification (word counts, TF-IDF) → Multinomial
# Binary features (word present/absent) → Bernoulli
# ============================================================


# ============================================================
# 5. PROS AND CONS
#
# PROS:
# SPEED: The FASTEST algorithm you will ever use —
#        milliseconds on massive datasets
#
# SMALL DATA: Works surprisingly well even if you only
#             have a few hundred rows
#
# SIMPLE: No complex hyperparameters like C or Gamma to
#         worry about
#
# CONS:
# THE "ZERO PROBABILITY" PROBLEM:
# If a specific feature value (like a specific word) never
# appeared in training, the model gives it a 0% probability
# and BREAKS the whole equation — because 0 × anything = 0,
# the entire prediction becomes 0 regardless of other
# features. Scikit-learn fixes this with "SMOOTHING"
# (also called Laplace Smoothing).
#
# BAD PROBABILITIES:
# Great at picking the RIGHT CLASS, but the "Probability %"
# it gives you is often NOT very reliable as a calibrated
# probability.
#
# VISUAL — zero probability problem:
#
# Training data never had the word "Blockchain"
# P("Blockchain" | Spam) = 0/N = 0
#
# New email: "FREE Blockchain WIN"
# P(Spam) = P(FREE|Spam) × P(Blockchain|Spam) × P(WIN|Spam)
#          = 0.9 × 0 × 0.85
#          = 0 ← COMPLETELY WRONG! ❌
#
# Laplace Smoothing (var_smoothing) adds a tiny value so
# no probability is ever exactly zero → fixes this.
# ============================================================


# ============================================================
# CODE
# ============================================================

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split

# 1. Split your data — NO StandardScaler needed!
# Naive Bayes doesn't use distances, it uses probabilities
# so scaling doesn't affect the result
X_train, X_test, y_train, y_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

# 2. Initialize the model
nb_model = GaussianNB()
# var_smoothing=1e-9 (default) → Laplace smoothing
# prevents zero probability problem

# 3. Train
nb_model.fit(X_train, y_train)

# 4. Predict
# So fast you won't even see a progress bar
score = nb_model.score(X_test, y_test)
print("Naive Bayes Accuracy Score:", score)

# For text classification use MultinomialNB:
# mnb = MultinomialNB(alpha=1.0)  ← alpha is smoothing

# For binary features use BernoulliNB:
# bnb = BernoulliNB(alpha=1.0)


# ============================================================
# NAIVE BAYES vs OTHER ALGORITHMS
#
# ┌──────────────────┬──────────────────┬───────────────────┐
# │                  │ Naive Bayes      │ SVM / RF          │
# ├──────────────────┼──────────────────┼───────────────────┤
# │ Speed            │ Milliseconds     │ Minutes           │
# │ Small data       │ Excellent        │ RF needs more data│
# │ Feature          │ Assumes complete │ Handles correlated│
# │ correlation      │ independence     │ features fine     │
# │ Probabilities    │ Unreliable %     │ More calibrated   │
# │ Best use case    │ Text/NLP, spam   │ Complex structured│
# │                  │ filtering        │ data              │
# └──────────────────┴──────────────────┴───────────────────┘
# ============================================================


# ============================================================
# SUMMARY
#
# If you are working with TEXT (NLP) or need a BASELINE
# MODEL in 2 seconds → use Naive Bayes.
#
# If your features are DEEPLY INTERCONNECTED (like in
# complex physics or finance) → stick to Gradient Boosting
# or SVM.
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Naive Bayes = probability-based, not geometry-based.
#    Uses Bayes' Theorem: combines prior probability with
#    likelihood of evidence to get posterior probability
#
# 2. "Naive" = assumes ALL features are INDEPENDENT.
#    Wrong assumption in theory, shockingly accurate in
#    practice (especially for text)
#
# 3. Three flavors:
#    Gaussian NB    → continuous numerical data
#    Multinomial NB → word counts/frequencies (NLP)
#    Bernoulli NB   → binary yes/no features
#
# 4. Zero Probability Problem: unseen feature values
#    kill the entire prediction (0 × anything = 0).
#    Fixed by Laplace Smoothing (var_smoothing / alpha)
#
# 5. NO scaling needed — probability based, not distance
#    based. This is a key difference from SVM and KNN.
#
# 6. INTERVIEW TRAP:
#    Q: "Why is Naive Bayes called 'Naive'?"
#    Right answer: "Because it makes the naive assumption
#    that ALL features are completely independent of each
#    other. In reality, features like weight and blood
#    pressure are correlated. But despite this wrong
#    assumption, Naive Bayes performs surprisingly well
#    especially for text classification and spam detection,
#    because even with correlated features, the combined
#    probability scores still push the correct class to
#    the highest rank most of the time."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Principal Component Analysis (PCA) in depth
#    Reducing dimensionality while preserving maximum
#    variance — perfect before running KNN or SVM on
#    high-dimensional data.
# ============================================================