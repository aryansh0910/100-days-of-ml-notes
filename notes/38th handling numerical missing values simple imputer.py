# ============================================================
# SIMPLE IMPUTER — UNIVARIATE IMPUTATION
# ============================================================

# ============================================================
# WHAT IS IT?
#
# SIMPLE IMPUTER IS A TECHNIQUE USED FOR UNIVARIATE ANALYSIS
# I.E THE COLUMN MISSING VALUES WILL BE FILLED ONLY WITH THE
# USE OF THE VALUES PRESENT IN THAT COLUMN. OTHER COLUMNS
# WILL NOT PLAY ANY ROLE.
#
# When Complete Case Analysis (CCA) deletes too much data,
# we use Imputation. SimpleImputer from Scikit-learn is the
# standard tool for filling in those "holes" (NaNs) with a
# specific strategy.
# ============================================================


# ============================================================
# STRATEGIES
#
# ┌───────────┬──────────────────────────┬─────────────────────────────┐
# │ Strategy  │ Logic                    │ When to use?                │
# ├───────────┼──────────────────────────┼─────────────────────────────┤
# │ Mean      │ Average of the column    │ Normal (Symmetric) dist.    │
# │ Median    │ Middle value             │ Skewed data, has outliers   │
# │ Constant  │ A specific number (0,-1) │ When missing means "Zero"   │
# └───────────┴──────────────────────────┴─────────────────────────────┘
#
# DISADVANTAGES OF FILLING MEAN AND MEDIAN:
# → CHANGES THE SHAPE OF THE DISTRIBUTION
# → CREATES SOME OUTLIERS
# → CAN CHANGE THE COVARIANCE BETWEEN COLUMNS
#
# HOW TO CHECK THE DAMAGE:
# → Boxplot before vs after → check for new outliers
# → df.corr() → check if correlations between columns changed
# → density plot before vs after → check distribution shape
# ============================================================


# ============================================================
# WHY SimpleImputer OVER pandas .fillna()?
#
# You might ask: "Can't I just use df.fillna(df.mean())?"
# You can, but SimpleImputer is better for ML because:
#
# THE "MEMORY" EFFECT:
# SimpleImputer learns the mean/median of TRAINING data
# and applies that SAME value to test data.
# fillna(df.mean()) recalculates mean on whatever df you
# pass — if you pass test data, it leaks test statistics.
#
# VISUAL:
#
# fillna() approach (WRONG):
# Train mean = 30 → fill train NaNs with 30
# Test mean  = 45 → fill test NaNs with 45  ← DATA LEAKAGE!
#
# SimpleImputer approach (CORRECT):
# imputer.fit(X_train)  → learns mean = 30
# imputer.transform(X_train) → fills with 30
# imputer.transform(X_test)  → fills with 30 ✅ same value!
#
# PIPELINE READY:
# Plugs directly into ColumnTransformer and Pipeline.
# fillna() can't do that.
# ============================================================


# ============================================================
# FILLING WITH A CONSTANT VALUE
#
# U CAN ALSO FILL THE MISSING VALUES WITH ANY TOLD VALUE
# LIKE 0, -1 AND OTHERS.
# U CAN USE fillna() IN CASE OF THIS.
# USING SIMPLE IMPUTER U HAVE TO USE strategy="constant"
# TO FILL THESE VALUES.
# ============================================================


# ============================================================
# END OF DISTRIBUTION IMPUTATION
#
# FINDING THE PERFECT VALUE TO FILL IS A VERY DIFFICULT TASK.
# SO U SEARCH FOR A PERFECT NUMBER TO FILL THE VALUES SO THAT
# THE DISTRIBUTION DOESN'T CHANGE, NEITHER THE COVARIANCE
# CHANGES — USING THE END OF DISTRIBUTION TECHNIQUE.
#
# HOW IT WORKS:
# Pick a value physically FAR AWAY from the rest of the data.
#
# For Normal Distribution  → Mean + (3 × Standard Deviation)
# For Skewed Distribution  → IQR proximity rule
#                            (far end of the whiskers)
#
# VISUAL:
#
# Normal distribution:
#
#        fill value
#            ↓
#  ___       |
# /   \      |
#/     \     |
#       \____↓___________
# ←─────────────────────→
# -3σ   μ               μ+3σ
#
# The fill value sits at the EXTREME END — far from all
# real data points. Model sees it as clearly "different."
#
# WHY USE THIS INSTEAD OF MEAN/MEDIAN?
#
# The goal isn't just to fill the hole — it's to FLAG
# the missingness as a SIGNAL.
#
# CAPTURES IMPORTANCE OF MISSING:
# If data is missing because a person was too sick to answer
# or a sensor failed during a massive surge, that
# "missingness" is information. Putting it at the end of
# the distribution tells the model: "this is a distinct,
# extreme case" — without needing an extra indicator column.
#
# U SPECIFICALLY USE THIS WHEN U KNOW THAT THE DATA IS
# NOT MISSING AT RANDOM.
#
# WHEN TO USE:
# → Linear Models or Tree-based models (Random Forest)
# → When you suspect missingness itself predicts the outcome
#
# WARNING — NEVER use this with distance-based algorithms:
# → KNN, SVM
# → The extreme value will massively distort distance
#   calculations and destroy your model's performance!
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# ============================================================
# SAMPLE DATA
# ============================================================

np.random.seed(42)
data = {
    'Age'    : [25, np.nan, 30, 35, np.nan, 28, 40, np.nan, 33, 27],
    'Salary' : [50000, 60000, np.nan, 55000, 70000,
                np.nan, 65000, 58000, np.nan, 52000],
    'Target' : [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['Age', 'Salary']]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# STRATEGY 1 — MEAN (for normal distributions)
# ============================================================

mean_imputer = SimpleImputer(strategy='mean')

# Golden rule — fit on train only
mean_imputer.fit(X_train)
print("Mean learned from train:", mean_imputer.statistics_)

X_train_mean = mean_imputer.transform(X_train)
X_test_mean  = mean_imputer.transform(X_test)
# Test data filled with TRAIN mean — no leakage ✅


# ============================================================
# STRATEGY 2 — MEDIAN (for skewed distributions)
# ============================================================

median_imputer = SimpleImputer(strategy='median')
median_imputer.fit(X_train)

X_train_median = median_imputer.transform(X_train)
X_test_median  = median_imputer.transform(X_test)


# ============================================================
# STRATEGY 3 — CONSTANT (filling with a fixed value)
# ============================================================

constant_imputer = SimpleImputer(strategy='constant',
                                  fill_value=0)
constant_imputer.fit(X_train)

X_train_const = constant_imputer.transform(X_train)
X_test_const  = constant_imputer.transform(X_test)


# ============================================================
# STRATEGY 4 — END OF DISTRIBUTION
# ============================================================

# Calculate fill value manually
train_df = pd.DataFrame(X_train, columns=['Age', 'Salary'])

age_fill_value = (train_df['Age'].mean() +
                  3 * train_df['Age'].std())

print(f"\nEnd of distribution fill value for Age: {age_fill_value:.2f}")

# Now use constant strategy with this calculated value
eod_imputer = SimpleImputer(strategy='constant',
                             fill_value=age_fill_value)
eod_imputer.fit(X_train[['Age']])

X_train_eod = eod_imputer.transform(X_train[['Age']])
X_test_eod  = eod_imputer.transform(X_test[['Age']])


# ============================================================
# CHECKING THE DAMAGE — PLOTS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Distribution before vs after mean imputation
axes[0].hist(X_train['Age'].dropna(), bins=10,
             color='red', alpha=0.5, label='Original')
axes[0].hist(X_train_mean[:, 0], bins=10,
             color='green', alpha=0.5, label='After Mean Fill')
axes[0].set_title('Distribution Before vs After')
axes[0].legend()

# Boxplot to check for new outliers
axes[1].boxplot([X_train['Age'].dropna(),
                 X_train_mean[:, 0]],
                labels=['Original', 'After Fill'])
axes[1].set_title('Boxplot — Check for New Outliers')

plt.tight_layout()
plt.show()

# U CAN ALSO CHECK COVARIANCE CHANGE:
print("\nCorrelation BEFORE imputation:")
print(X_train.corr())

X_train_df = pd.DataFrame(X_train_mean,
                           columns=['Age', 'Salary'])
print("\nCorrelation AFTER imputation:")
print(X_train_df.corr())
# If numbers changed significantly → imputation affected
# the relationship between columns


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. SimpleImputer = UNIVARIATE → uses only that column
#    to fill its own missing values. Other columns ignored.
#
# 2. Strategy selection:
#    Normal distribution  → mean
#    Skewed / outliers    → median
#    Missing means zero   → constant (fill_value=0)
#    Missingness is signal→ end of distribution
#
# 3. ALWAYS fit on train, transform both:
#    fillna() recalculates every time → data leakage
#    SimpleImputer remembers train statistics → safe ✅
#
# 4. DISADVANTAGES of mean/median filling:
#    → Changes distribution shape
#    → Can create outliers
#    → Changes covariance between columns
#    Always verify with boxplot + corr() + density plot!
#
# 5. END OF DISTRIBUTION:
#    Fill value = Mean + 3×Std (normal) or IQR far end (skewed)
#    Use ONLY when missingness itself is a signal
#    NEVER use with KNN or SVM — distorts distances badly!
#
# 6. INTERVIEW TRAP:
#    Q: "Why not just use df.fillna(df.mean())?"
#    Right answer: "fillna() recalculates on whatever data
#    you pass — if you pass test data, you leak test stats
#    into your pipeline. SimpleImputer fits on train once
#    and applies the same value to test. Also fits into
#    sklearn Pipelines, fillna() doesn't."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → KNN Imputer
#    Instead of filling with a single statistic, KNN Imputer
#    finds the K most similar rows and uses their values
#    to fill the missing entry. Smarter, but slower.
# ============================================================