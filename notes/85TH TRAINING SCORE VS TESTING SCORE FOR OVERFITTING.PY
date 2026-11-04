# ============================================================
# MODEL COMPARISON — OVERFITTING GAP ANALYSIS
# ============================================================

# ============================================================
# WHAT IS IT?
#
# HOW TO READ THESE RESULTS (THE "RULE OF THUMB"):
#
# LOW GAP (Small number):
# Your model is HONEST. What it learned in training, it
# can actually do in the real world.
#
# HIGH GAP (Large number):
# Your model is a "MEMORIZER." It's "cheating" by
# memorizing the training answers and will FAIL on new data.
#
# NEGATIVE GAP:
# This is RARE, but it means your Test Score is actually
# HIGHER than your Train Score. This usually just means
# you got "LUCKY" with a very easy Test Set or your
# dataset is very small.
#
# VISUAL — reading the gap:
#
# Model A: Train=0.99, Test=0.62 → Gap=0.37 ❌ OVERFITTING
# Model B: Train=0.88, Test=0.85 → Gap=0.03 ✅ HONEST
# Model C: Train=0.72, Test=0.74 → Gap=-0.02 ⚠️ LUCKY/SMALL DATA
#
# ┌──────────────────────────────────────────────────────┐
# │ Gap < 0.05  → Great generalization ✅               │
# │ Gap 0.05-0.10 → Slight overfitting, acceptable ⚠️   │
# │ Gap > 0.10  → Overfitting, needs fixing ❌           │
# └──────────────────────────────────────────────────────┘
# ============================================================


# ============================================================
# CODE — COMPARING ALL MODELS AT ONCE
# ============================================================

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 1. Define your list of models
models = [
    ('SVM',
     SVC(kernel='rbf', C=1, probability=True)),
    ('Naive Bayes',
     GaussianNB()),
    ('Decision Tree',
     DecisionTreeClassifier(max_depth=5)),
    ('Random Forest',
     RandomForestClassifier(n_estimators=100, max_depth=5)),
    ('XGBoost',
     XGBClassifier(n_estimators=100, learning_rate=0.1,
                   max_depth=3))
]

# 2. Run the loop
print("=" * 50)
for name, model in models:
    # Train
    model.fit(X_train, y_train)

    # Get scores directly using .score()
    train_score = model.score(X_train, y_train)
    test_score  = model.score(X_test, y_test)

    # Calculate the difference
    gap = train_score - test_score

    # Flag overfitting automatically
    if gap > 0.10:
        status = "❌ OVERFITTING"
    elif gap < 0:
        status = "⚠️  LUCKY/SMALL DATA"
    else:
        status = "✅ GOOD"

    print(f"Model          : {name}")
    print(f"Train Score    : {train_score:.4f}")
    print(f"Test Score     : {test_score:.4f}")
    print(f"Overfitting Gap: {gap:.4f}  {status}")
    print("-" * 50)


# ============================================================
# WHAT TO DO WHEN GAP IS HIGH
#
# Decision Tree overfitting → reduce max_depth
# Random Forest overfitting → reduce max_depth,
#                             increase min_samples_leaf
# XGBoost overfitting       → reduce learning_rate,
#                             increase gamma,
#                             reduce max_depth
# SVM overfitting           → reduce C
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Gap = Train Score - Test Score
#    Low gap → generalizes well
#    High gap → memorizing training data (overfitting)
#    Negative gap → got lucky or very small dataset
#
# 2. Always compare ALL models together — the best test
#    score with the lowest gap wins
#
# 3. High train score alone means NOTHING — what matters
#    is how it performs on data it has NEVER seen
#
# 4. INTERVIEW TRAP:
#    Q: "Your model has 99% training accuracy but 65%
#        test accuracy. What is happening and how do
#        you fix it?"
#    Right answer: "The model is overfitting — memorizing
#    training data instead of learning generalizable
#    patterns. Gap = 34% which is very high. Fix depends
#    on the model: for trees reduce max_depth or increase
#    min_samples_leaf; for XGBoost increase gamma and
#    reduce learning_rate; for SVM reduce C. Also consider
#    getting more training data or using cross-validation
#    instead of a single train-test split to get a more
#    reliable estimate of true performance."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Cross Validation in depth
#    More reliable than a single train-test split —
#    tests the model on every fold of the data.
# ============================================================