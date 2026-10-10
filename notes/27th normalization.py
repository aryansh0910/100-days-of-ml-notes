# ============================================================
# NORMALIZATION — MIN-MAX SCALING
# ============================================================

# ============================================================
# WHAT IS IT?
#
# Normalization (specifically Min-Max Scaling) is the go-to
# technique when you need your data to fit into a strictly
# bounded range — usually 0 to 1.
#
# While Standardization (Z-score) centers data around 0
# with no upper limit, Normalization "squashes" everything
# so that the smallest value becomes exactly 0 and the
# largest becomes exactly 1.
#
# THE FORMULA:
# ┌─────────────────────────────────────────────┐
# │                                             │
# │         x_i - x_min                        │
# │   x  =  ────────────                       │
# │         x_max - x_min                      │
# │                                             │
# │  x_i   = original value                    │
# │  x_min = minimum value in that column      │
# │  x_max = maximum value in that column      │
# │                                             │
# └─────────────────────────────────────────────┘
#
# ANALOGY (Desi Style 🏏):
# Think of cricket scores in a match.
# Lowest score = 0 runs, Highest score = 150 runs.
# Min-Max scaling converts every score so that:
#   → 0 runs   becomes 0.0
#   → 150 runs becomes 1.0
#   → 75 runs  becomes 0.5 (exactly halfway)
# Everyone is now compared on a 0-to-1 report card!
#
# Another way to think about it:
# Normalization asks: "Where does this value sit
# between the WORST and the BEST?"
# ============================================================


# ============================================================
# VISUAL MENTAL MODEL
#
# BEFORE Normalization:
#
#  Age column:
#  |----|----|----|----|----|----|
#  18   30   42   54   66   78   90
#  (range = 72, numbers all over the place)
#
# AFTER Normalization:
#
#  |----|----|----|----|----|----|
#  0.0  0.2  0.4  0.6  0.8  1.0
#  (everything squashed into 0 → 1)
#
# THE OUTLIER PROBLEM (most important visual):
#
# Without outlier:
# Ages: 20, 22, 24, 26, 28, 30
# After scaling: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0  ✅ spread out
#
# WITH outlier (one person aged 100):
# Ages: 20, 22, 24, 26, 28, 30, 100
# After scaling: 0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 1.0
#                |___________________________________|
#                All normal people squashed here!  ❌
#
# The 100-year-old HIJACKS the entire scale!
# ============================================================


# ============================================================
# WHEN TO USE NORMALIZATION?
#
# You should pick Normalization over Standardization in
# these specific scenarios:
#
# 1. UNKNOWN DISTRIBUTION:
#    When you don't know if your data follows a bell curve
#    (Normal Distribution). Normalization doesn't care about
#    the shape of your data — it just uses min and max.
#
# 2. DISTANCE-BASED ALGORITHMS:
#    Highly effective for KNN and Neural Networks
#    (which often prefer inputs strictly between 0 and 1).
#
# 3. IMAGE PROCESSING:
#    Pixel values are normalized from [0, 255] → [0, 1].
#    This is the MOST common real-world use of MinMaxScaler!
#
# QUICK DECISION GUIDE:
# ┌──────────────────────────────────────────────────────┐
# │  Got outliers?          → Use StandardScaler         │
# │  Need strict 0-1 range? → Use MinMaxScaler           │
# │  Neural Networks / CNN? → MinMaxScaler (0-1)         │
# │  Don't know which?      → Start with StandardScaler  │
# └──────────────────────────────────────────────────────┘
# ============================================================


# ============================================================
# THE MAJOR WEAKNESS: OUTLIERS
#
# This is a favorite interview question in the CampusX series.
# Normalization is EXTREMELY sensitive to outliers.
#
# Imagine you have 100 people aged 20–30, but one person
# is 100 years old.
#
# → The person who is 100 becomes 1.0
# → The people aged 20–30 all get "squashed" into a tiny
#   range like 0.20 to 0.30
# → The model loses the ability to distinguish between a
#   21-year-old and a 29-year-old because they look almost
#   identical after scaling!
#
# WHY DOES THIS HAPPEN?
# Because the formula uses x_max in the denominator.
# One giant outlier makes x_max huge →
# denominator becomes huge →
# all normal values become tiny fractions → squashed!
#
# INTERVIEW TRAP ⚠️:
# Q: "Does MinMaxScaler handle outliers well?"
# Wrong answer: "Yes, because it scales everything to 0-1"
# Right answer: "NO — it's the MOST outlier-sensitive scaler
#                because outliers directly control x_min and
#                x_max, which define the entire scale."
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# ============================================================
# CREATING SAMPLE DATA
# ============================================================

np.random.seed(42)
data = {
    'Age'    : np.random.randint(18, 80, 1000),
    'Salary' : np.random.randint(20000, 150000, 1000),
    'Target' : np.random.randint(0, 2, 1000)
}
df = pd.DataFrame(data)

X = df[['Age', 'Salary']]
y = df['Target']

print("=" * 55)
print("BEFORE NORMALIZATION")
print("=" * 55)
print(f"Age    → Min: {X['Age'].min()}, Max: {X['Age'].max()}")
print(f"Salary → Min: {X['Salary'].min()}, Max: {X['Salary'].max()}")
print()


# ============================================================
# THE GOLDEN RULE — SAME AS STANDARDIZATION
# FIT ON TRAIN ONLY, TRANSFORM BOTH
# ============================================================
#
# Exact same rule as StandardScaler — don't let test data
# influence the scaler. If you fit on test data, the scaler
# learns the min/max of test set → DATA LEAKAGE ❌
#
# CORRECT WORKFLOW:
# ┌──────────────────────────────────────────────────────┐
# │                                                      │
# │  scaler.fit(X_train)           → learn min, max      │
# │  scaler.transform(X_train)     → scale train         │
# │  scaler.transform(X_test)      → scale test          │
# │                                                      │
# │  OR shortcut:                                        │
# │  scaler.fit_transform(X_train) → fit AND transform   │
# │                                                      │
# │  ❌ NEVER: scaler.fit(X_test)  → DATA LEAKAGE!       │
# │                                                      │
# └──────────────────────────────────────────────────────┘

# STEP 1 → Split FIRST, then scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 2 → Create MinMaxScaler object
scaler = MinMaxScaler()

# STEP 3 → Fit ONLY on training data
# Learns x_min and x_max from training data ONLY
scaler.fit(X_train)

print("=" * 55)
print("SCALER LEARNED FROM TRAINING DATA:")
print("=" * 55)
print(f"Min learned : {scaler.data_min_}")
print(f"Max learned : {scaler.data_max_}")
print()

# STEP 4 → Transform BOTH train and test
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ============================================================
# VERIFYING THE OUTPUT
# ============================================================

X_train_df = pd.DataFrame(X_train_scaled,
                           columns=['Age', 'Salary'])

print("=" * 55)
print("AFTER NORMALIZATION (Training Data)")
print("=" * 55)
print(f"Age    → Min: {X_train_df['Age'].min():.4f},"
      f" Max: {X_train_df['Age'].max():.4f}")
print(f"Salary → Min: {X_train_df['Salary'].min():.4f},"
      f" Max: {X_train_df['Salary'].max():.4f}")
print()
print("✅ Both columns now between 0 and 1!")
print("✅ Shape of distribution is PRESERVED!")


# ============================================================
# DEMONSTRATING THE OUTLIER PROBLEM
# ============================================================

# Normal ages: 20-30
normal_ages = np.array([[20], [22], [24], [26], [28], [30]])

# Same ages BUT with one outlier
outlier_ages = np.array([[20], [22], [24], [26], [28], [30], [100]])

scaler_normal  = MinMaxScaler()
scaler_outlier = MinMaxScaler()

scaled_normal  = scaler_normal.fit_transform(normal_ages)
scaled_outlier = scaler_outlier.fit_transform(outlier_ages)

print("=" * 55)
print("OUTLIER EFFECT DEMONSTRATION")
print("=" * 55)
print("WITHOUT outlier:")
for age, scaled in zip(normal_ages.flatten(),
                       scaled_normal.flatten()):
    print(f"  Age {age} → {scaled:.3f}")

print("\nWITH outlier (100-year-old added):")
for age, scaled in zip(outlier_ages.flatten(),
                       scaled_outlier.flatten()):
    print(f"  Age {age:3d} → {scaled:.3f}")

print("\n⚠️  See how ages 20-30 are now squashed into")
print("   0.0 to 0.125 range because of one outlier!")


# ============================================================
# NORMALIZATION vs STANDARDIZATION — FULL COMPARISON
# ============================================================
#
# ┌─────────────────┬──────────────────┬───────────────────┐
# │                 │ MinMaxScaler     │ StandardScaler    │
# │                 │ (Normalization)  │ (Standardization) │
# ├─────────────────┼──────────────────┼───────────────────┤
# │ Output range    │ Strictly 0 to 1  │ No fixed range    │
# │ Formula uses    │ Min and Max       │ Mean and Std      │
# │ Outlier effect  │ VERY sensitive   │ Less sensitive    │
# │ Distribution    │ Shape preserved  │ Shape preserved   │
# │ Best for        │ Neural Networks  │ Most algorithms   │
# │                 │ Image data       │ Unknown dist.     │
# │                 │ Bounded ranges   │                   │
# └─────────────────┴──────────────────┴───────────────────┘
#
# SIMPLE RULE TO REMEMBER:
# No outliers + need 0-1  → MinMaxScaler
# Outliers present        → StandardScaler (safer choice)
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Formula: x = (x_i - x_min) / (x_max - x_min)
#    Result: every value strictly between 0 and 1
#
# 2. WHEN to use:
#    → Unknown distribution
#    → Neural Networks / Deep Learning
#    → Image processing (255 → 1)
#    → Any algorithm needing strict 0-1 range
#
# 3. BIGGEST WEAKNESS — Outliers:
#    One outlier becomes 1.0 and HIJACKS the entire scale
#    All normal values get squashed into a tiny range
#    Always remove outliers BEFORE using MinMaxScaler!
#
# 4. THE GOLDEN RULE is the same as StandardScaler:
#    fit() on TRAIN only → transform BOTH train and test
#    NEVER fit on test data → DATA LEAKAGE ❌
#
# 5. Shape of distribution is PRESERVED (same as StandardScaler)
#    Skewed stays skewed, bell curve stays bell curve
#
# 6. INTERVIEW TRAP:
#    "MinMaxScaler handles outliers since everything is 0-1"
#    → WRONG. 0-1 range doesn't mean outlier-safe.
#    → Outliers DEFINE the 0 and 1 endpoints, squashing
#      all other values in between!
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Encoding Categorical Data
#    Converting text categories like "Male/Female" or
#    "City Names" into numbers that ML models understand.
#    Label Encoding vs One Hot Encoding — when to use which,
#    and the multicollinearity trap to avoid!
# ============================================================