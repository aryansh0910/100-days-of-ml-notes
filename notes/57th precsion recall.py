# ============================================================
# PRECISION, RECALL, F1-SCORE
# ============================================================

# ============================================================
# 1. PRECISION — THE "CERTAINTY" METRIC
#
# Precision asks: "Of all the customers the model
# PREDICTED would churn, how many ACTUALLY did?"
#
# GOAL: Minimize False Positives (Type 1 Errors)
#
# BUSINESS IMPACT: High precision means you aren't wasting
# money sending "Save" discounts to people who were going
# to stay anyway.
#
# FORMULA:
# ┌─────────────────────────────────────────────┐
# │                                             │
# │                  TP                          │
# │   Precision  =  ──────────                  │
# │                 TP + FP                      │
# │                                             │
# └─────────────────────────────────────────────┘
#
# VISUAL — what precision is asking:
#
# Model predicted "Churn" for 10 customers:
# ┌──────────────────────────────────────────┐
# │ ● ● ● ● ● ● ● ○ ○ ○                       │
# │ ●=actually churned (TP)   ○=didn't (FP)   │
# └──────────────────────────────────────────┘
# 7 actually churned, 3 were wrong (false alarms)
# Precision = 7 / (7+3) = 0.70
# "When the model says Churn, it's right 70% of the time"
# ============================================================


# ============================================================
# 2. RECALL — THE "COVERAGE" METRIC
#
# Recall (also called Sensitivity) asks: "Of all the
# customers who ACTUALLY churned, how many did the model
# MANAGE TO CATCH?"
#
# GOAL: Minimize False Negatives (Type 2 Errors)
#
# BUSINESS IMPACT: High recall means you are catching
# almost every "at-risk" customer. This is usually the
# PRIORITY in churn projects because losing a customer is
# more expensive than sending a redundant coupon.
#
# FORMULA:
# ┌─────────────────────────────────────────────┐
# │                                             │
# │                  TP                          │
# │   Recall  =  ──────────                      │
# │              TP + FN                          │
# │                                             │
# └─────────────────────────────────────────────┘
#
# VISUAL — what recall is asking:
#
# 10 customers ACTUALLY churned in reality:
# ┌──────────────────────────────────────────┐
# │ ● ● ● ● ● ● ● ✗ ✗ ✗                       │
# │ ●=model caught them (TP)  ✗=model missed (FN)│
# └──────────────────────────────────────────┘
# Model caught 7 out of 10 actual churners
# Recall = 7 / (7+3) = 0.70
# "The model catches 70% of customers who actually churn"
# ============================================================


# ============================================================
# 3. THE TUG-OF-WAR — THE TRADE-OFF
#
# There is a natural tension between these two.
#
# IF YOU WANT 100% RECALL:
# You can just predict "Everyone will churn." You'll catch
# every single leaver, but your Precision will be terrible
# because you'll have thousands of False Alarms.
#
# IF YOU WANT 100% PRECISION:
# You only predict "Churn" for the one person you are
# absolutely, 100% certain about. Your precision is
# perfect, but your Recall will be terrible because you
# missed all the other "maybe" cases.
#
# VISUAL — the two extremes:
#
# 100% Recall strategy (predict EVERYONE churns):
# Caught ALL 10 actual churners ✅ (Recall = 100%)
# But also flagged 990 people who were staying ❌
# Precision = 10/1000 = 1% (terrible)
#
# 100% Precision strategy (only flag the ONE certain case):
# That 1 flagged person definitely churned ✅ (Precision=100%)
# But missed the other 9 actual churners ❌
# Recall = 1/10 = 10% (terrible)
#
# Neither extreme is useful — you need a BALANCE based on
# what the business actually cares about more.
# ============================================================


# ============================================================
# 4. F1-SCORE — THE "BALANCED" METRIC
#
# The F1-Score is the HARMONIC MEAN of Precision and
# Recall. It gives you a single number that represents the
# balance between the two.
#
# USE THIS WHEN: You want a balance between catching
# leavers and not wasting budget. It PUNISHES extreme
# values — if either Precision or Recall is very low, the
# F1-Score will tank.
#
# FORMULA:
# ┌─────────────────────────────────────────────┐
# │                                             │
# │           Precision × Recall                │
# │   F1  =  2 × ──────────────────              │
# │              Precision + Recall              │
# │                                             │
# └─────────────────────────────────────────────┘
#
# WHY HARMONIC MEAN, NOT REGULAR AVERAGE?
# Regular average is forgiving of extremes:
# Precision=1.0, Recall=0.01 → average = 0.505 (looks okay!)
#
# Harmonic mean punishes extremes:
# Precision=1.0, Recall=0.01 → F1 = 0.0198 (correctly tanks!)
#
# The harmonic mean is dragged down hard by the smaller
# value — exactly what you want when one metric is
# secretly terrible.
# ============================================================


# ============================================================
# QUICK REFERENCE TABLE
#
# ┌─────────────┬───────────┬─────────────────────────────┐
# │ Metric      │ Focus     │ Question it Answers          │
# ├─────────────┼───────────┼─────────────────────────────┤
# │ Precision   │ Quality   │ "Is my Churn alarm reliable?"│
# │ Recall      │ Quantity  │ "Am I missing any Churners?" │
# │ F1-Score    │ Balance   │ "Is the model overall        │
# │             │           │  healthy?"                   │
# └─────────────┴───────────┴─────────────────────────────┘
# ============================================================


# ============================================================
# 6. CODING IT IN PYTHON
#
# You don't need to calculate these by hand. Scikit-learn's
# classification_report is the best way to see everything
# at once.
# ============================================================

from sklearn.metrics import classification_report

# This prints a table with Precision, Recall, and F1 for
# BOTH classes (0 and 1)
print(classification_report(y_test, y_pred,
                            target_names=['Stay', 'Churn']))


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Precision = TP / (TP+FP) → "When I say churn, am I right?"
#    High precision → few false alarms
#
# 2. Recall = TP / (TP+FN) → "Did I catch all the churners?"
#    High recall → few missed churners
#
# 3. There's a TRADE-OFF — improving one often hurts the
#    other. Business priority decides which matters more.
#
# 4. For Churn → Recall usually matters MORE (missing a
#    churner costs more than a wasted discount)
#
# 5. F1-Score = harmonic mean of Precision and Recall —
#    punishes extreme imbalance between the two, unlike a
#    regular average which is too forgiving
#
# 6. classification_report() gives Precision, Recall, F1
#    for every class in one call — no manual calculation needed
#
# 7. INTERVIEW TRAP:
#    Q: "Why use harmonic mean for F1 instead of a simple
#        average of Precision and Recall?"
#    Right answer: "A simple average is forgiving — a model
#    with Precision=1.0 and Recall=0.01 would average to
#    0.505, which looks decent but the model is actually
#    useless (catches almost nothing). Harmonic mean punishes
#    extreme imbalance much more harshly, correctly reflecting
#    that the model has a serious problem."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → ROC-AUC Curve
#    Visualizing the trade-off between True Positive Rate
#    and False Positive Rate across all possible thresholds.
# ============================================================