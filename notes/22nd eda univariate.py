# ============================================================
# UNIVARIATE ANALYSIS — EDA LEVEL 1
# ============================================================

# ============================================================
# WHAT IS IT?
# EDA (Exploratory Data Analysis) is where you stop being
# a data "collector" and start being a data "detective."
#
# EDA = using visualizations and statistical summaries to:
# → Find patterns hidden in the data
# → Spot anomalies and errors
# → Check assumptions before modeling
#
# It is usually divided into THREE levels of difficulty:
# Level 1 → Univariate Analysis   (one variable at a time)
# Level 2 → Bivariate Analysis    (two variables together)
# Level 3 → Multivariate Analysis (many variables together)
#
# THIS FILE COVERS LEVEL 1 — UNIVARIATE ANALYSIS
# ============================================================


# ============================================================
# UNIVARIATE ANALYSIS
# ============================================================
#
# THE INDEPENDENT ANALYSIS OF EVERY COLUMN IS KNOWN
# AS UNIVARIATE ANALYSIS.
#
# You examine ONE variable at a time to understand:
# → What does this column look like?
# → Is it normally distributed or skewed?
# → Are there outliers?
# → Which category dominates?
#
# DATA IS GENERALLY DIVIDED INTO TWO TYPES:
# ┌─────────────────────────────────────────────┐
# │ 1. NUMERICAL   → Age, Salary, Price, Fare   │
# │ 2. CATEGORICAL → Genre, City, Survived      │
# └─────────────────────────────────────────────┘
# Each type needs DIFFERENT tools for analysis.
# ============================================================


# ============================================================
# IMPORTS AND DATA LOADING
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# WE WILL USE THE TITANIC DATASET FOR THIS
df = pd.read_csv("22nd_train_data.csv")
print("=" * 55)
print("FIRST LOOK AT THE DATA")
print("=" * 55)
print(df.head(1))
print()
print(df.describe())
print()


# ============================================================
# PART A — CATEGORICAL DATA ANALYSIS
# ============================================================
#
# For columns like Genre, City, Education, or Survived
# you want to see the FREQUENCY of each group.
# How many rows belong to each category?
#
# STATISTICAL SUMMARY → value_counts()
# ─────────────────────────────────────
# df['column'].value_counts() tells you EXACTLY how many
# rows belong to each category.
# Mode = the most frequent category.
#
# ANALOGY:
# Like counting how many students chose each stream
# in your college — Science, Commerce, Arts.
# You just want to know who has the most students.

print("=" * 55)
print("CATEGORICAL — value_counts() for Survived")
print("=" * 55)
print(df["Survived"].value_counts())
print(f"\nMode (most frequent): {df['Survived'].mode()[0]}")
print()

# ============================================================
# VISUAL TOOLS FOR CATEGORICAL DATA
# ============================================================

plt.figure(figsize=(12, 10))

# ── COUNTPLOT (Bar Chart) ────────────────────────────────────
# The most common way to see WHICH CATEGORY DOMINATES.
# Seaborn automatically counts values and draws bars.
# WE WILL PLOT THE COUNTPLOT TO CHECK WHICH OF THE
# SURVIVED PEOPLE IS MORE OR THE DEAD PEOPLE

plt.subplot(2, 2, 1)
sns.countplot(data=df, x="Survived")
plt.title("Survived — Countplot")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
# HOW TO READ:
# Tall bar = more people in that category
# Short bar = fewer people in that category
# Big height difference = IMBALANCED data ⚠️

# ── PIE CHART ────────────────────────────────────────────────
# IF U NEED THAT IN PERCENTAGE U CAN USE PIE CHART
# Great for seeing the PERCENTAGE SHARE of each category.
# Example: "60% of Titanic passengers did NOT survive"
# autopct = "%.2f" → shows percentage with 2 decimal places

plt.subplot(2, 2, 2)
df["Survived"].value_counts().plot(kind="pie", autopct="%.2f%%",
                                    labels=["Did Not Survive",
                                            "Survived"],
                                    colors=["coral", "steelblue"])
plt.title("Survived — Pie Chart (% Share)")
plt.ylabel("")  # removes the default "Survived" y-label
# HOW TO READ:
# Bigger slice = dominant category
# Very unequal slices = imbalanced dataset
# Example: 61.6% vs 38.4% → mild imbalance ⚠️


# ============================================================
# PART B — NUMERICAL DATA ANALYSIS
# ============================================================
#
# For columns like Age, Salary, Fare, Price
# you want to understand the DISTRIBUTION of values.
# How spread out is the data? Is it skewed?
#
# STATISTICAL SUMMARY → describe()
# ─────────────────────────────────
# THE 5-NUMBER SUMMARY:
# Mean / Median → Where is the CENTER?
#   If Mean > Median → data is RIGHT SKEWED  📈
#   If Mean < Median → data is LEFT SKEWED   📉
#   If Mean ≈ Median → data is NORMAL        🔔
#
# Std (Standard Deviation) → How SPREAD OUT are the numbers?
#   Low std  → values are clustered close to the mean
#   High std → values are spread far from the mean
#
# Min / Max → What are the BOUNDARIES?
#   Check for impossible values (negative age, age=200 etc.)

print("=" * 55)
print("NUMERICAL — describe() for Age")
print("=" * 55)
print(df["Age"].describe())
print(f"\nMean   : {df['Age'].mean():.2f}")
print(f"Median : {df['Age'].median():.2f}")
if df["Age"].mean() > df["Age"].median():
    print("→ Mean > Median = RIGHT SKEWED data 📈")
else:
    print("→ Mean < Median = LEFT SKEWED data 📉")
print()

# ============================================================
# VISUAL TOOLS FOR NUMERICAL DATA
# ============================================================

# ── DISTPLOT / KDE PLOT ──────────────────────────────────────
# FIRST THING U CAN DO ON NUMERICAL DATA IS CREATE BINS
# AND SEE WHERE THE VALUES LIE THE MOST.
# A smooth version of a histogram showing the DENSITY
# of the data and the SHAPE of the distribution.
#
# SHAPES TO LOOK FOR:
# 🔔 Normal (Bell Curve)  → mean ≈ median, symmetric
# 📈 Right Skewed         → long tail on the RIGHT
#                           most values are LOW
#                           mean is pulled RIGHT by outliers
# 📉 Left Skewed          → long tail on the LEFT
#                           most values are HIGH
# 🏔️ Bimodal             → two peaks = two different groups
#                           in your data

plt.subplot(2, 2, 3)
sns.histplot(df["Age"].dropna(), kde=True, color="steelblue")
# NOTE: sns.distplot() is deprecated in newer seaborn versions
# Use sns.histplot(kde=True) instead — does the same thing!
plt.title("Age — Distribution Plot (KDE)")
plt.xlabel("Age")
plt.ylabel("Density")
print(df["Age"].head())
print()

# ── BOXPLOT ──────────────────────────────────────────────────
# THE KING OF OUTLIER DETECTION!
# U SHOULD KNOW HOW TO READ THE BOX PLOT
# VALUES LYING OUTSIDE THE LINE ARE THE OUTLIERS
# SO IT HELPS U IDENTIFY THE OUTLIERS
#
# HOW TO READ A BOXPLOT:
# ┌─────────────────────────────────────────────────────┐
# │                                                     │
# │    o ← OUTLIER (dot outside whiskers)               │
# │    |                                                │
# │  ──┤ ← UPPER WHISKER (Q3 + 1.5×IQR)               │
# │    │                                                │
# │  ┌─┴─┐                                             │
# │  │   │ ← Q3 (75th percentile)                      │
# │  ├───┤ ← MEDIAN (50th percentile) ← middle line    │
# │  │   │ ← Q1 (25th percentile)                      │
# │  └─┬─┘                                             │
# │    │                                                │
# │  ──┤ ← LOWER WHISKER (Q1 - 1.5×IQR)               │
# │    |                                                │
# │    o ← OUTLIER (dot outside whiskers)               │
# └─────────────────────────────────────────────────────┘
#
# The BOX = middle 50% of your data (IQR)
# The LINE inside box = median
# The WHISKERS = range of "normal" data
# The DOTS outside whiskers = POTENTIAL OUTLIERS ⚠️
#
# For Fare: expect lots of outliers on the upper end
# (some first class passengers paid extremely high fares)

plt.subplot(2, 2, 4)
plt.boxplot(df["Fare"].dropna())
plt.title("Fare — Boxplot (Outlier Detection)")
plt.ylabel("Fare (£)")
# HOW TO READ THIS OUTPUT:
# Many dots above the upper whisker = RIGHT SKEWED
# Most passengers paid low fares
# A few paid extremely high fares → those are outliers


plt.tight_layout()
plt.savefig('univariate_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("Univariate EDA plots saved!\n")


# ============================================================
# QUICK DECISION GUIDE — WHICH TOOL TO USE?
# ============================================================
#
# NUMERICAL DATA:
# ┌──────────────────┬────────────────────────────────────┐
# │ Tool             │ Use it for                         │
# ├──────────────────┼────────────────────────────────────┤
# │ describe()       │ Quick stats summary                │
# │ Histogram/KDE    │ See distribution shape             │
# │ Boxplot          │ Detect outliers                    │
# └──────────────────┴────────────────────────────────────┘
#
# CATEGORICAL DATA:
# ┌──────────────────┬────────────────────────────────────┐
# │ Tool             │ Use it for                         │
# ├──────────────────┼────────────────────────────────────┤
# │ value_counts()   │ Count each category                │
# │ Countplot        │ Visualize frequency of categories  │
# │ Pie Chart        │ See percentage share               │
# └──────────────────┴────────────────────────────────────┘
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Univariate = ONE variable at a time
#    THE INDEPENDENT ANALYSIS OF EVERY COLUMN
#
# 2. Data types decide your tools:
#    Numerical   → describe(), KDE plot, Boxplot
#    Categorical → value_counts(), Countplot, Pie Chart
#
# 3. Reading skewness from describe():
#    Mean > Median → Right Skewed (long tail right)
#    Mean < Median → Left Skewed  (long tail left)
#    Mean ≈ Median → Normal distribution
#
# 4. Boxplot is THE KING of outlier detection
#    Dots outside whiskers = potential outliers
#    Decide: real data or error?
#
# 5. sns.distplot() is DEPRECATED in newer seaborn
#    Use sns.histplot(kde=True) instead
#    Does the exact same thing, just updated syntax
#
# 6. Countplot = seaborn automatically counts for you
#    Pie chart  = use .plot(kind='pie', autopct='%.2f%%')
#
# 7. Always check for IMBALANCED data in categorical columns
#    Very unequal countplot/pie = imbalanced dataset
#    Must handle before classification tasks
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Bivariate Analysis (EDA Level 2)
#    Looking at the relationship between TWO variables.
#    Scatter plots, box plots by category, and heatmaps
#    to understand how features interact with each other
#    and which ones actually affect the target variable.
# ============================================================