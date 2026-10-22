# ============================================================
# CONFUSION MATRIX, ACCURACY, TYPE 1 & TYPE 2 ERRORS
# ============================================================

# ============================================================
# 1. THE CONFUSION MATRIX
#
# The Confusion Matrix is a 2×2 table that breaks down your
# model's "Hits" and "Misses."
#
# TRUE NEGATIVE means that person DOESN'T HAVE heart disease
# and our computer SAYS THE SAME
#
# TRUE POSITIVE means that person HAS heart disease and our
# computer SAYS THE SAME
#
# FALSE NEGATIVE means that person HAS heart disease and our
# computer SAYS IT DOESN'T (VERY BAD)
#
# FALSE POSITIVE means that person DOESN'T HAVE heart
# disease and our computer SAYS IT DOES
#
#                      Predicted: No (0)      Predicted: Yes (1)
#                    ┌──────────────────┬──────────────────┐
# Actual: No (0)     │  True Negative   │  False Positive  │
#                    │      (TN)        │      (FP)        │
#                    ├──────────────────┼──────────────────┤
# Actual: Yes (1)    │  False Negative  │  True Positive   │
#                    │      (FN)        │      (TP)        │
#                    └──────────────────┴──────────────────┘
# ============================================================


# ============================================================
# 2. ACCURACY SCORE — THE TRAP
#
# Accuracy is simply:
#
# ┌─────────────────────────────────────────────┐
# │                                             │
# │             TP + TN                         │
# │  Accuracy = ───────────────                 │
# │             TP + TN + FP + FN                │
# │                                             │
# └─────────────────────────────────────────────┘
#
# ACCURACY ==> (NO. OF CORRECT PREDICTIONS) /
#              (TOTAL NO. OF PREDICTIONS)
#
# IT IS VERY MISLEADING WHEN UR DATA IS IMBALANCED. LIKE
# WHEN THE PERSON HAVING HEART DISEASE IS 10 AND THE
# PERSONS HAVING NO DISEASE IS 990, THEN IN THIS CASE U
# CAN'T BELIEVE UR ACCURACY SCORE EVEN IF IT'S 99.9%
#
# THE PROBLEM WITH ACCURACY SCORE IS THAT IT DOESN'T TELL
# YOU WHAT TYPE OF PROBLEM IS PRESENT IN YOUR MODEL — LIKE
# REAL PLACEMENT HAPPENED AND MODEL SAID IT DIDN'T, AND
# PLACEMENT DIDN'T HAPPEN BUT IN REALITY IT DID — ACCURACY
# TREATS BOTH MISTAKES THE SAME.
#
# WHY IT'S DANGEROUS FOR CHURN:
# If 95% of your customers stay and only 5% churn, a
# "dumb" model that predicts "No Churn" for EVERYONE will
# be 95% accurate. It sounds great, but it failed to catch
# a SINGLE person leaving!
#
# VISUAL — the dumb model trap:
#
# 1000 customers: 950 stay, 50 churn
#
# Dumb model predicts "No Churn" for ALL 1000:
# TN = 950, TP = 0, FP = 0, FN = 50
#
# Accuracy = (950 + 0) / 1000 = 95% ← looks AMAZING ✅
#
# But it caught ZERO churners. Completely useless for the
# actual business goal! ❌
# ============================================================


# ============================================================
# 3. TYPE 1 AND TYPE 2 ERRORS
#
# This is where the business strategy comes in. Every
# "Miss" has a different COST.
# ============================================================


# ============================================================
# TYPE 1 ERROR: FALSE POSITIVE (The "False Alarm")
#
# THE SITUATION: The model says a customer WILL churn, but
# they actually intended to stay.
#
# BUSINESS RESULT: You send them a 50% discount code they
# didn't need. You lost a little bit of profit, but you
# kept the customer happy.
#
# COST: Relatively LOW — wasted money, but no real damage.
# ============================================================


# ============================================================
# TYPE 2 ERROR: FALSE NEGATIVE (The "Miss")
#
# THE SITUATION: The model says a customer WILL stay, but
# they actually churn.
#
# BUSINESS RESULT: You did NOTHING. The customer left and
# is now with your competitor. This is usually much more
# expensive!
#
# COST: Much HIGHER — you lost the customer entirely with
# no chance to intervene.
#
# VISUAL — comparing the two mistakes for Churn:
#
# Type 1 (False Positive):
# Predicted: Will Churn → Actual: Stayed
# Action taken: sent discount → Cost: small (wasted discount)
#
# Type 2 (False Negative):
# Predicted: Will Stay → Actual: Churned
# Action taken: nothing → Cost: HIGH (lost customer entirely)
#
# This is WHY accuracy alone is dangerous — it doesn't
# distinguish between a "wasted discount" mistake and a
# "lost the customer forever" mistake. Both count the same
# in the accuracy formula, but they're NOT equally bad for
# the business!
# ============================================================


# ============================================================
# CODE
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Sample predictions
y_actual = np.array([0,0,0,0,0,0,0,0,0,1, 1,1,1,1,1,
                     0,0,0,0,0])  # imbalanced: 5 churn, 15 stay
y_pred   = np.array([0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,
                     0,0,0,0,0])  # dumb model: predicts all stay

cm = confusion_matrix(y_actual, y_pred)
print("Confusion Matrix:")
print(cm)

print(f"\nAccuracy: {accuracy_score(y_actual, y_pred)}")
# Will be high (75%) even though it caught ZERO churners

# Visualize
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Predicted No', 'Predicted Yes'],
           yticklabels=['Actual No', 'Actual Yes'])
plt.title('Confusion Matrix')
plt.show()


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Confusion Matrix has 4 cells: TN, TP, FN, FP
#    Remember: First letter = was prediction correct (True/
#    False), Second word = what was predicted (Positive/
#    Negative)
#
# 2. Accuracy = (TP+TN) / (TP+TN+FP+FN)
#    DANGEROUS on imbalanced data — a model that always
#    predicts the majority class gets high accuracy while
#    being completely useless
#
# 3. Type 1 Error (False Positive) = false alarm
#    Type 2 Error (False Negative) = missed detection
#
# 4. The COST of each error type depends on the business
#    context — for churn, Type 2 (missing a churner) is
#    usually much more expensive than Type 1
#
# 5. For imbalanced datasets, NEVER trust accuracy alone —
#    use Precision, Recall, F1-score instead (which look
#    specifically at how well the minority class is caught)
#
# 6. INTERVIEW TRAP:
#    Q: "Your model has 99% accuracy on a fraud detection
#        dataset. Is it a good model?"
#    Wrong: "Yes, 99% accuracy is excellent"
#    Right: "Not necessarily — fraud datasets are usually
#    highly imbalanced (maybe 1% fraud, 99% legitimate).
#    A model predicting 'not fraud' for everyone would also
#    get 99% accuracy while catching zero fraud cases.
#    Need to check the confusion matrix, recall, and
#    precision to know if it's actually useful."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Precision, Recall, and F1-Score
#    Metrics that actually account for class imbalance and
#    tell you how well your model catches the minority class.
# ============================================================