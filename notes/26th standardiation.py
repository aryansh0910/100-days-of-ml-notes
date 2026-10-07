# ============================================================
# STANDARDIZATION — FEATURE SCALING
# ============================================================

# ============================================================
# WHAT IS IT?
# Standardization (also called Z-Score Normalization) is
# the most common technique for Feature Scaling.
#
# Standardization transforms your data so that it has:
# → Mean (μ)              = 0
# → Standard Deviation (σ) = 1
#
# THE FORMULA:
# ┌─────────────────────────────────────────────┐
# │                                             │
# │         x_i - μ                             │
# │   x  =  ───────                             │
# │            σ                                │
# │                                             │
# │  x_i = original value                       │
# │  μ   = mean (average) of the column         │
# │  σ   = standard deviation (spread)          │
# │                                             │
# └─────────────────────────────────────────────┘
#
# ANALOGY:
# Imagine two students — one scores in a 0-100 system
# and another in a 0-10 system. Directly comparing
# their raw scores is unfair. Standardization converts
# BOTH to the same scale so comparison is fair.
# ============================================================


# ============================================================
# WHY DO WE NEED IT?
# ============================================================
#
# Imagine you have:
# Age            → 0 to 80
# Estimated Salary → 20,000 to 1,50,000
#
# If you use a model like KNN or Logistic Regression,
# the model calculates "DISTANCE" between data points.
# Because salary numbers are HUGE, the model will think
# Salary is more important than Age just because the
# numbers are bigger — even if Age matters more!
#
# EXAMPLE:
# Person A → Age=25,  Salary=50,000
# Person B → Age=26,  Salary=50,001
#
# Distance WITHOUT scaling:
# = √((26-25)² + (50001-50000)²)
# = √(1 + 1) = √2 ≈ 1.41
# Age difference = 1, Salary difference = 1 → EQUAL weight ✅
#
# BUT with Age 0-80 and Salary 0-150000:
# Person A → Age=25,  Salary=20,000
# Person B → Age=60,  Salary=20,001
#
# Distance = √((60-25)² + (20001-20000)²)
#          = √(1225 + 1) ≈ 35
# Salary barely contributed! Age DOMINATED unfairly ❌
#
# Standardization puts BOTH on the same scale → fair model!
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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
print("BEFORE STANDARDIZATION")
print("=" * 55)
print(f"Age    → Mean: {X['Age'].mean():.2f},"
      f" Std: {X['Age'].std():.2f},"
      f" Range: {X['Age'].min()}-{X['Age'].max()}")
print(f"Salary → Mean: {X['Salary'].mean():.2f},"
      f" Std: {X['Salary'].std():.2f},"
      f" Range: {X['Salary'].min()}-{X['Salary'].max()}")
print()


# ============================================================
# THE GOLDEN RULE — FIT ON TRAIN, TRANSFORM BOTH
# ============================================================
#
# USING SCALER.FIT MEANS YOU ARE LETTING THE MODEL
# UNDERSTAND THE DATA AND SET THE MEAN AND STD DEVIATION.
# YOU JUST USE FIT ON THE TRAIN DATA BUT YOU TRANSFORM
# BOTH THE TEST AS WELL AS THE TRAIN DATA.
# BECAUSE FIT ONLY UNDERSTANDS THE VALUES, NOT TRANSFORMS.
#
# NEVER use .fit() on your test data!
# You only fit on TRAINING data so that your model treats
# the test data as "unseen" information.
#
# If you fit on test data → you are "CHEATING" by letting
# the model know the average of the test set.
# This is called DATA LEAKAGE — a critical mistake!
#
# THE CORRECT WORKFLOW:
# ┌──────────────────────────────────────────────────────┐
# │                                                      │
# │  scaler.fit(X_train)           → learn from train   │
# │  scaler.transform(X_train)     → scale train        │
# │  scaler.transform(X_test)      → scale test         │
# │                                                      │
# │  OR shortcut for train only:                         │
# │  scaler.fit_transform(X_train) → fit AND transform  │
# │                                                      │
# │  ❌ NEVER: scaler.fit(X_test)  → DATA LEAKAGE!      │
# │  ❌ NEVER: scaler.fit_transform(X_test)              │
# │                                                      │
# └──────────────────────────────────────────────────────┘

# STEP 1 → Split FIRST, then scale
# (Never scale before splitting — that causes data leakage!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 2 → Create scaler object
scaler = StandardScaler()

# STEP 3 → fit ONLY on training data
# This learns the mean and std from training data ONLY
scaler.fit(X_train)

print("=" * 55)
print("SCALER LEARNED FROM TRAINING DATA:")
print("=" * 55)
print(f"Mean learned  : {scaler.mean_}")
print(f"Scale learned : {scaler.scale_}")
print()

# STEP 4 → transform BOTH train and test
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# SHORTCUT: fit_transform on train only
# X_train_scaled = scaler.fit_transform(X_train)  ← same result
# X_test_scaled  = scaler.transform(X_test)       ← transform only


# ============================================================
# VERIFYING THE OUTPUT
# ============================================================

X_train_df = pd.DataFrame(X_train_scaled,
                           columns=['Age', 'Salary'])

print("=" * 55)
print("AFTER STANDARDIZATION (Training Data)")
print("=" * 55)
print(f"Age    → Mean: {X_train_df['Age'].mean():.4f},"
      f" Std: {X_train_df['Age'].std():.4f}")
print(f"Salary → Mean: {X_train_df['Salary'].mean():.4f},"
      f" Std: {X_train_df['Salary'].std():.4f}")
print()
print("✅ Both columns now have Mean ≈ 0 and Std ≈ 1!")
print("✅ Both on same scale → model treats them fairly!")


# ============================================================
# EFFECT ON DISTRIBUTION — THE MOST IMPORTANT CONCEPT
# ============================================================
#
# HERE IS THE MOST IMPORTANT THING TO REMEMBER
# FOR YOUR INTERVIEWS:
#
# Standardization does NOT change the SHAPE of distribution!
#
# If your data was SKEWED   → it will STAY skewed
# If it was a BELL CURVE    → it STAYS a bell curve
#
# The ONLY thing that changes is the SCALE on the X-axis
# It will now CENTER around 0.
#
# REMEMBER: SCALING DOESN'T HAVE ANY EFFECT ON OUTLIERS
# SO YOU HAVE TO REMOVE THEM YOURSELF BEFORE SCALING!
#
# WHY? Because StandardScaler uses MEAN and STD to scale.
# Outliers PULL the mean and inflate the std.
# Result → all normal values get squished together
# → scaling becomes ineffective
# Always handle outliers BEFORE scaling!

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Original Age distribution
axes[0, 0].hist(X_train['Age'], bins=30,
                color='steelblue', edgecolor='black')
axes[0, 0].set_title('Age — BEFORE Scaling')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

# Scaled Age distribution
axes[0, 1].hist(X_train_df['Age'], bins=30,
                color='coral', edgecolor='black')
axes[0, 1].set_title('Age — AFTER Scaling')
axes[0, 1].set_xlabel('Z-Score')
axes[0, 1].set_ylabel('Frequency')

# Original Salary distribution
axes[1, 0].hist(X_train['Salary'], bins=30,
                color='steelblue', edgecolor='black')
axes[1, 0].set_title('Salary — BEFORE Scaling')
axes[1, 0].set_xlabel('Salary')
axes[1, 0].set_ylabel('Frequency')

# Scaled Salary distribution
axes[1, 1].hist(X_train_df['Salary'], bins=30,
                color='coral', edgecolor='black')
axes[1, 1].set_title('Salary — AFTER Scaling')
axes[1, 1].set_xlabel('Z-Score')
axes[1, 1].set_ylabel('Frequency')

plt.suptitle('Standardization — Shape PRESERVED, Scale CHANGED',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('standardization_effect.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Distribution plots saved!")
print("NOTICE: Shape is identical before and after!")
print("ONLY the x-axis values changed → now centered at 0")


# ============================================================
# STANDARDIZATION vs NORMALIZATION — QUICK COMPARISON
# ============================================================
#
# ┌─────────────────┬──────────────────┬───────────────────┐
# │                 │ StandardScaler   │ MinMaxScaler      │
# │                 │ (Standardization)│ (Normalization)   │
# ├─────────────────┼──────────────────┼───────────────────┤
# │ Output range    │ No fixed range   │ 0 to 1            │
# │ Mean after      │ 0                │ Not fixed         │
# │ Std after       │ 1                │ Not fixed         │
# │ Outlier effect  │ Less sensitive   │ Very sensitive    │
# │ Best for        │ Normal dist data │ Bounded range     │
# │ Use with        │ Most algorithms  │ Neural Networks   │
# └─────────────────┴──────────────────┴───────────────────┘
#
# IN PRACTICE:
# Start with StandardScaler → works well for most cases
# Switch to MinMaxScaler → if model needs 0-1 range (ANN)


# ============================================================
# ALGORITHMS — NEED SCALING OR NOT?
# ============================================================
#
# ✅ MUST SCALE (distance or gradient based):
#    → KNN           (uses Euclidean distance)
#    → K-Means       (uses Euclidean distance)
#    → SVM           (uses distance from hyperplane)
#    → PCA           (variance based → scale sensitive)
#    → ANN / DL      (gradient descent → scale sensitive)
#    → Logistic Reg  (gradient descent)
#    → Linear Reg    (gradient descent)
#
# ❌ DON'T NEED SCALING (tree based = split based):
#    → Decision Trees  (splits on values, not distances)
#    → Random Forest   (ensemble of trees)
#    → XGBoost         (gradient boosted trees)
#    → CatBoost        (tree based)
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Standardization formula: x = (x_i - μ) / σ
#    Result: Mean = 0, Std = 1 for every column
#
# 2. WHY we need it:
#    Distance based models treat large numbers as more important
#    Scaling puts ALL features on equal footing
#
# 3. THE GOLDEN RULE — most important for interviews:
#    fit()           → learns mean and std (TRAIN ONLY)
#    transform()     → applies the scaling (TRAIN AND TEST)
#    fit_transform() → shortcut for train only
#    NEVER fit on test data = DATA LEAKAGE ❌
#
# 4. Standardization does NOT change distribution SHAPE
#    Skewed stays skewed, bell curve stays bell curve
#    ONLY the x-axis scale changes → centered at 0
#
# 5. SCALING DOES NOT REMOVE OUTLIERS
#    Handle outliers BEFORE scaling, not after!
#    Outliers affect mean and std → bad scaling
#
# 6. StandardScaler vs MinMaxScaler:
#    StandardScaler → most common, less outlier sensitive
#    MinMaxScaler   → use when you need strict 0-1 range
#
# 7. Tree based models (Random Forest, XGBoost) do NOT
#    need scaling — they split on values not distances
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Encoding Categorical Data
#    Converting text categories like "Male/Female" or
#    "City Names" into numbers that ML models understand.
#    Label Encoding vs One Hot Encoding — when to use which,
#    and the multicollinearity trap to avoid!
# ============================================================