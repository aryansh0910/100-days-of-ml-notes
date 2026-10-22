# ============================================================
# SOFTMAX REGRESSION (MULTINOMIAL LOGISTIC REGRESSION)
# ============================================================

# ============================================================
# WHAT IS IT?
#
# LOGISTIC REGRESSION IS ONLY USED FOR BINARY CLASSIFICATION
# LIKE JUST YES OR NO. U HAVE TO USE SOFTMAX REGRESSION FOR
# MORE THAN TWO CATEGORIES. ALSO KNOWN AS MULTINOMIAL
# LOGISTIC REGRESSION.
#
# If Logistic Regression is for "Yes/No" (Binary
# Classification), Softmax Regression (also known as
# Multinomial Logistic Regression) is its BIG BROTHER for
# when you have three or more categories.
#
# While standard Logistic Regression is for "Yes/No"
# choices, Softmax Regression is designed for 3 OR MORE
# classes. It is the standard way to handle multiclass
# classification.
#
# BINARY: Predicting if a customer will Churn (1) or
#         Stay (0).
#
# MULTICLASS (Softmax): Predicting if a customer will
# choose the "Basic Plan," "Premium Plan," or "Elite Plan."
# ============================================================


# ============================================================
# HOW SOFTMAX EXTENDS SIGMOID
#
# Sigmoid handles 2 classes — gives ONE probability, and
# the other class is just (1 - that probability).
#
# Softmax handles N classes — gives N probabilities, one
# per class, and ALL of them must add up to exactly 1.
#
# VISUAL — Sigmoid vs Softmax outputs:
#
# SIGMOID (binary — Churn or Stay):
# P(Churn) = 0.72
# P(Stay)  = 1 - 0.72 = 0.28
# (only need to compute ONE value, other is automatic)
#
# SOFTMAX (3 classes — Basic, Premium, Elite):
# P(Basic)   = 0.10
# P(Premium) = 0.65   ← highest, so this is the prediction
# P(Elite)   = 0.25
# Sum = 0.10 + 0.65 + 0.25 = 1.00 ✅ always sums to 1
#
# The class with the HIGHEST probability becomes the final
# prediction — here, "Premium Plan."
# ============================================================


# ============================================================
# CODE
#
# JUST PASS THE multi_class PARAMETER IN THE
# LogisticRegression, THEN IT WILL PERFORM SOFTMAX
# REGRESSION.
# ============================================================

from sklearn.linear_model import LogisticRegression

# Use 'multinomial' for Softmax
# Use 'lbfgs' or 'saga' solvers as they support multiclass
model = LogisticRegression(multi_class='multinomial')

# Training
model.fit(X_train_scaled, y_train)

# Get probabilities for all classes
probs = model.predict_proba(X_test_scaled)
print("Probabilities for each class:")
print(probs[:5])
# Each row sums to 1.0, one column per class

# Get final class prediction
predictions = model.predict(X_test_scaled)
print("\nPredictions:", predictions[:5])
# Picks the class with the HIGHEST probability per row