# ============================================================
# UNDERSTANDING THE DATA — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# ============================================================
# WHAT IS IT?
# Before building ANY model, you must deeply understand
# your data. Skipping this step is like a doctor
# prescribing medicine without examining the patient first.
#
# EDA = Exploratory Data Analysis
# The process of INVESTIGATING your data to find:
# → What does it look like?
# → What patterns exist?
# → What problems need fixing?
#
# A great ML engineer spends MORE time here than
# anywhere else. This is where intuition is built.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (Car Dekho used as example)
# df = pd.read_csv('car_dekho.csv')

# For demonstration we create a sample dataset
data = {
    'Age'       : [25, 30, 35, 28, 200, 22, 29, 31, None, 27],
    'Salary'    : [30000, 45000, 60000, 35000, 50000,
                   28000, 42000, 55000, 38000, None],
    'Education' : ['UG', 'PG', 'PG', 'UG', 'PhD',
                   'UG', 'UG', 'PG', 'UG', 'PG'],
    'Churned'   : [0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
}
df = pd.DataFrame(data)


# ============================================================
# STAGE 1 — THE "FIRST DATE" (Basic Inspection)
# ============================================================
#
# Before you plot any graphs, you need to see the
# "SKELETON" of your data.
# Like meeting someone for the first time —
# you notice the basics before going deep.
#
# THREE ESSENTIAL COMMANDS:
#
# ── df.head() / df.sample() ──────────────────────────────────
# Look at the actual values.
# Are they messy? Are there NaN values?
# df.head()   → always shows first 5 rows (can be biased)
# df.sample() → shows random rows (more honest view)

print("=" * 55)
print("df.head() — First 5 rows")
print("=" * 55)
print(df.head())
print()

print("=" * 55)
print("df.sample(5) — Random 5 rows")
print("=" * 55)
print(df.sample(5, random_state=42))
print()

# ── df.info() ────────────────────────────────────────────────
# Check the DATA TYPES.
# Are numbers being treated as strings?
# Are there missing values? (check Non-Null count)
# This is your FIRST clue about data quality problems.

print("=" * 55)
print("df.info() — Data Types + Missing Values")
print("=" * 55)
df.info()
print()

# HOW TO READ df.info() OUTPUT:
# Column     Non-Null Count   Dtype
# Age        9 non-null       float64  ← 1 missing value!
# Salary     9 non-null       float64  ← 1 missing value!
# Education  10 non-null      object   ← no missing values
#
# Non-Null Count < total rows = MISSING VALUES EXIST ❌

# ── df.describe() ────────────────────────────────────────────
# Get the STATISTICS.
# Look for mean, std, min, max for potential outliers.
# Example: if "Age" column has max = 200 → something is WRONG

print("=" * 55)
print("df.describe() — Statistical Summary")
print("=" * 55)
print(df.describe())
print()

# HOW TO READ df.describe() OUTPUT:
# count → how many non-null values
# mean  → average value
# std   → how spread out the values are
# min   → smallest value (check for negatives that shouldn't be)
# 25%   → 25th percentile
# 50%   → median
# 75%   → 75th percentile
# max   → largest value (check for impossibly large values)
#
# ALERT SIGNALS:
# Age max = 200  → impossible age = data error        ❌
# Salary min = -5000 → negative salary = data error   ❌
# Huge gap between 75% and max → outlier exists       ⚠️


# ============================================================
# STAGE 2 — UNIVARIATE ANALYSIS
# ============================================================
#
# This means looking at ONE variable at a time.
# You want to understand the "DISTRIBUTION" of each column.
# How is the data spread out? Is it balanced or skewed?
#
# ANALOGY:
# Like studying each player individually before
# analyzing how the team performs together.

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ── Numerical Columns → Histogram / Distplot ─────────────────
# Question: Is the data "Normal" (bell curve) or skewed?
#
# Normal Distribution  → mean ≈ median ≈ mode (symmetric)
# Right Skewed         → long tail on the right (most values low)
# Left Skewed          → long tail on the left (most values high)
#
# WHY IT MATTERS:
# Many ML algorithms assume normal distribution.
# Skewed data may need log transformation before training.

axes[0].hist(df['Age'].dropna(), bins=8, color='steelblue',
             edgecolor='black')
axes[0].set_title('Age Distribution')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Frequency')

axes[1].hist(df['Salary'].dropna(), bins=8, color='coral',
             edgecolor='black')
axes[1].set_title('Salary Distribution')
axes[1].set_xlabel('Salary')

# ── Categorical Columns → Bar Chart / Pie Chart ──────────────
# Question: Is one category dominating the data?
# Example: In a churn dataset, are 90% of users staying?
# This is "IMBALANCED DATA" and must be handled carefully.
#
# Imbalanced data → model learns to always predict majority
# Example: 90% Not Churn → model says "Not Churn" always
#          gets 90% accuracy but is completely useless! ❌

churn_counts = df['Churned'].value_counts()
axes[2].bar(['Not Churned (0)', 'Churned (1)'],
            churn_counts.values,
            color=['steelblue', 'coral'],
            edgecolor='black')
axes[2].set_title('Churn Distribution (Balanced?)')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.savefig('univariate_analysis.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("Univariate plots saved!\n")


# ============================================================
# STAGE 3 — BIVARIATE & MULTIVARIATE ANALYSIS
# ============================================================
#
# This is WHERE THE MAGIC HAPPENS.
# You look at the RELATIONSHIP between variables.
# How does one column affect another?
#
# ANALOGY:
# Like analyzing how two cricket players perform TOGETHER
# vs analyzing each one alone. Team chemistry matters!

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── Scatter Plot → Two Numerical Columns ─────────────────────
# Perfect for seeing how two numbers RELATE.
# Example: Does "Age" increase as "Salary" increases?
# If points go up-right → POSITIVE correlation
# If points go down-right → NEGATIVE correlation
# If points are random → NO correlation

axes[0].scatter(df['Age'], df['Salary'],
                color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_title('Age vs Salary (Scatter Plot)')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Salary')

# ── Box Plot → Category vs Number ────────────────────────────
# Best for comparing a CATEGORY against a NUMBER.
# Example: "Salary" vs "Education Level"
# Helps you spot OUTLIERS instantly.
#
# HOW TO READ A BOX PLOT:
# The box     → middle 50% of data (IQR)
# Middle line → median
# Whiskers    → range of normal data
# Dots outside whiskers → OUTLIERS ⚠️

edu_groups = [df[df['Education'] == edu]['Salary'].dropna()
              for edu in ['UG', 'PG', 'PhD']]
axes[1].boxplot(edu_groups, labels=['UG', 'PG', 'PhD'],
                patch_artist=True)
axes[1].set_title('Salary vs Education (Box Plot)')
axes[1].set_xlabel('Education Level')
axes[1].set_ylabel('Salary')

# ── Heatmap → Correlation Matrix ─────────────────────────────
# Use df.corr() with a heatmap to see which features
# are strongly LINKED to each other.
#
# VALUES RANGE FROM -1 to +1:
# +1.0  → perfect positive correlation (move together)
#  0.0  → no correlation (independent)
# -1.0  → perfect negative correlation (move opposite)
#
# 💡 TIP: If two features are 99% correlated,
# you might only need ONE of them for your model!
# Keeping both = redundant information = slower model

corr_matrix = df[['Age', 'Salary', 'Churned']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f',
            cmap='coolwarm', ax=axes[2],
            linewidths=0.5)
axes[2].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('bivariate_analysis.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("Bivariate plots saved!\n")


# ============================================================
# STAGE 4 — IDENTIFYING "THE BIG THREE" PROBLEMS
# ============================================================
#
# While understanding your data, you are specifically
# HUNTING for these three issues.
# Find them early → fix them before modeling.

print("=" * 55)
print("THE BIG THREE PROBLEMS")
print("=" * 55)

# ── PROBLEM 1: MISSING VALUES ─────────────────────────────────
# Do you DROP the rows or FILL them with mean/median?
#
# RULE OF THUMB:
# < 5% missing  → safe to drop the rows
# 5-30% missing → fill with mean (numerical) or
#                 mode (categorical)
# > 30% missing → consider dropping the entire column

print("\n1. MISSING VALUES")
print("-" * 40)
print(df.isnull().sum())
print(f"\nTotal missing: {df.isnull().sum().sum()}")

missing_percent = (df.isnull().sum() / len(df)) * 100
print("\nMissing Percentage:")
print(missing_percent[missing_percent > 0])

# ── PROBLEM 2: OUTLIERS ───────────────────────────────────────
# Are the extreme values REAL data or ERRORS?
#
# REAL OUTLIER    → A billionaire's income in a salary dataset
#                   Keep it — it's genuine data
# ERROR OUTLIER   → A negative age or Age = 200
#                   Remove it — it's a data entry mistake
#
# HOW TO DETECT:
# Method 1 → IQR (InterQuartile Range)
# Method 2 → Z-Score
# Method 3 → Box Plot (visual)
#
# IQR METHOD:
# Q1 = 25th percentile
# Q3 = 75th percentile
# IQR = Q3 - Q1
# Lower bound = Q1 - 1.5 × IQR
# Upper bound = Q3 + 1.5 × IQR
# Anything outside these bounds = OUTLIER

print("\n2. OUTLIERS (IQR Method)")
print("-" * 40)
Q1  = df['Age'].quantile(0.25)
Q3  = df['Age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Age'] < lower_bound) |
              (df['Age'] > upper_bound)]
print(f"Age Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Normal range: {lower_bound:.1f} to {upper_bound:.1f}")
print(f"Outliers found:\n{outliers[['Age']]}")

# ── PROBLEM 3: DUPLICATES ─────────────────────────────────────
# Are the same records repeating and BIASING your results?
# Duplicates make the model think some data points are
# more important than they actually are.

print("\n3. DUPLICATES")
print("-" * 40)
print(f"Total duplicate rows: {df.duplicated().sum()}")
print(f"Shape before drop   : {df.shape}")

df_clean = df.drop_duplicates()
print(f"Shape after drop    : {df_clean.shape}")


# ============================================================
# QUICK EDA CHECKLIST
# ============================================================
#
# Run these EVERY TIME you get a new dataset:
#
# [ ] df.head() + df.sample()  → see actual data
# [ ] df.info()                → check dtypes + missing
# [ ] df.describe()            → check stats + outliers
# [ ] df.isnull().sum()        → count missing values
# [ ] df.duplicated().sum()    → count duplicates
# [ ] Histogram for numericals → check distribution
# [ ] Bar chart for categoricals → check balance
# [ ] Scatter plots            → check relationships
# [ ] Correlation heatmap      → check feature links
# [ ] Box plots                → visualize outliers
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Always start with the "First Date" trio:
#    df.head() → df.info() → df.describe()
#    These three alone tell you 80% of what you need to know
#
# 2. Univariate = one variable at a time
#    Numerical   → Histogram (check for skew)
#    Categorical → Bar/Pie chart (check for imbalance)
#
# 3. Bivariate / Multivariate = relationships between variables
#    Two numbers     → Scatter Plot
#    Category + Number → Box Plot (outliers visible!)
#    All features    → Correlation Heatmap
#
# 4. If two features are 99% correlated → keep only ONE
#    Redundant features slow down the model with no benefit
#
# 5. THE BIG THREE problems to always hunt for:
#    Missing Values → drop or fill (mean/median/mode)
#    Outliers       → real data or error? decide carefully
#    Duplicates     → always drop them
#
# 6. Imbalanced data is a TRAP:
#    90% one class → model predicts that class always
#    Gets high accuracy but is completely useless
#    Handle with SMOTE, class_weight, or oversampling
#
# 7. df.sample() > df.head() for honest data inspection
#    head() always shows the first rows which may not
#    be representative of the full dataset
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Feature Engineering
#    Creating new meaningful columns from existing ones,
#    selecting the most useful features,
#    and using techniques like PCA to reduce dimensions —
#    the "secret sauce" that separates good models
#    from great ones.
# ============================================================