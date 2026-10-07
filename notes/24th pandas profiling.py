# ============================================================
# PANDAS PROFILING — THE MAGIC WAND OF EDA
# ============================================================

# ============================================================
# WHAT IS IT?
# Pandas Profiling (now officially renamed to ydata-profiling)
# is treated as the "Magic Wand" of EDA.
# It is a very powerful tool to study the given data
# so you should learn how to read the pandas profiling report.
#
# THE PROBLEM IT SOLVES:
# Manually writing sns.histplot, df.describe(), sns.heatmap
# and all other EDA plots takes HOURS of coding.
# Pandas Profiling generates a COMPREHENSIVE HTML report
# containing ALL of these analyses with just ONE line of code.
#
# ANALOGY:
# Manual EDA = cooking a full meal from scratch
# Pandas Profiling = ordering a complete thali that has
# everything already prepared — roti, dal, sabzi, rice,
# dessert — all in one plate, instantly delivered!
# ============================================================


# ============================================================
# INSTALLATION
# ============================================================
#
# pip install ydata-profiling
#
# NOTE: The old package name was "pandas-profiling"
# It has been officially renamed to "ydata-profiling"
# Both do the same thing, ydata is just the newer version.
# Always use: from ydata_profiling import ProfileReport
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from ydata_profiling import ProfileReport
import pandas as pd

# Load the Titanic dataset
df = pd.read_csv("22nd_train_data.csv")


# ============================================================
# WHAT DOES THE REPORT INCLUDE?
# ============================================================
#
# When you run it, you get a CLICKABLE, INTERACTIVE WEBPAGE
# that summarizes everything about your dataset:
#
# 1. OVERVIEW
#    → Number of rows and columns
#    → Total missing cells (and % missing)
#    → Number of duplicate rows
#    → Memory usage of the dataset
#
# 2. VARIABLE INSIGHTS (for EVERY column automatically)
#    → Distribution plot (histogram + KDE)
#    → Mean, min, max, standard deviation
#    → Percentage of zeros
#    → Number of unique values
#    → Top frequent values for categorical columns
#
# 3. INTERACTIONS
#    → Automatically generated scatter plots
#      between every pair of numerical variables
#    → No need to write pairplot manually!
#
# 4. CORRELATIONS
#    → Heatmaps for THREE types of correlations:
#      Pearson  → linear relationships (most common)
#      Spearman → monotonic relationships (rank based)
#      Kendall  → for small datasets or ordinal data
#
# 5. MISSING VALUES
#    → Detailed charts showing WHERE data is "empty"
#    → Example: Cabin column in Titanic has 77% missing
#    → Matrix view shows PATTERNS of missingness
#    → Helps you decide: drop column or fill values?
#
# 6. WARNINGS (The Most Useful Section!)
#    → "High Correlation" between two features
#    → "High Cardinality" (too many unique values)
#    → "Imbalanced" target variable
#    → "Zeros" (column has many zero values)
#    → These warnings directly tell you what to fix
#       before training your ML model!
# ============================================================


# ============================================================
# BASIC USAGE — GENERATE THE REPORT
# ============================================================

# STEP 1 — Create the Profile Report object
# This line does ALL the analysis internally
prof = ProfileReport(df)
# THIS WILL CREATE A FULL REPORT OBJECT.
# NOW WE WILL SAVE THIS REPORT IN OUR SYSTEM
# IN HTML FORMAT SO WE CAN VIEW IT IN THE BROWSER.

# STEP 2 — Save the report as an HTML file
prof.to_file(output_file="24th_output_file.html")
# → Open this file in any browser (Chrome, Firefox)
# → You get a fully interactive, clickable report
# → Click on any column to expand its full analysis
# → Much better than static matplotlib plots!

print("✅ Profile report saved as 24th_output_file.html")
print("   Open it in your browser to explore!")


# ============================================================
# ADVANCED USAGE — DIFFERENT MODES
# ============================================================

# ── MODE 1: MINIMAL (for large datasets) ─────────────────────
# WHEN NOT TO USE NORMAL PROFILING:
# If your file has MILLIONS of rows, Pandas Profiling will
# FREEZE YOUR COMPUTER because it tries to calculate
# everything at once.
# Solution → use minimal=True for huge datasets.

prof_minimal = ProfileReport(df, minimal=True)
prof_minimal.to_file(output_file="profile_minimal.html")
print("✅ Minimal report saved as profile_minimal.html")

# ── MODE 2: TITLE + EXPLORATIVE ──────────────────────────────
# Add a custom title and enable deeper analysis

prof_full = ProfileReport(
    df,
    title="Titanic Dataset — Full EDA Report",
    explorative=True        # enables deeper correlation analysis
)
prof_full.to_file(output_file="profile_full.html")
print("✅ Full explorative report saved as profile_full.html")

# ── MODE 3: VIEW IN JUPYTER NOTEBOOK ─────────────────────────
# If you are using Jupyter, you can display it inline
# without saving to a file:
#
# prof.to_notebook_iframe()
#
# This renders the full interactive report directly
# inside your Jupyter cell — very convenient!


# ============================================================
# THE THREE CORRELATION TYPES — EXPLAINED
# ============================================================
#
# Profiling gives you THREE correlation heatmaps.
# Here's when each one matters:
#
# PEARSON (most common):
# → Measures LINEAR relationships
# → Example: as Age increases, Fare increases linearly
# → Best for normally distributed numerical data
# → Value range: -1 to +1
#
# SPEARMAN (rank based):
# → Measures MONOTONIC relationships (not just linear)
# → Works even if relationship is curved
# → Better when data is SKEWED or has outliers
# → Value range: -1 to +1
#
# KENDALL (for small/ordinal data):
# → Based on CONCORDANT vs DISCORDANT pairs
# → More robust on small datasets
# → Used for ordinal columns (ranks, ratings)
# → Value range: -1 to +1
#
# IN PRACTICE:
# For most ML projects → look at PEARSON first
# If data is skewed or has outliers → check SPEARMAN too
# If you see big differences between them → investigate why
# ============================================================


# ============================================================
# THE WARNINGS SECTION — MOST USEFUL FOR ML
# ============================================================
#
# This is the section that directly helps your ML model.
# Pandas Profiling explicitly warns you about:
#
# ⚠️  HIGH CORRELATION
#     Two features are 90%+ correlated
#     → Consider dropping one to avoid redundancy
#     → Keeping both = multicollinearity problem in ML
#
# ⚠️  HIGH CARDINALITY
#     A categorical column has too many unique values
#     Example: "Name" column has 891 unique values
#     → Useless for ML, should be dropped
#     → Or extract meaningful info from it (feature engineering)
#
# ⚠️  IMBALANCED
#     One category dominates (e.g., 90% class 0, 10% class 1)
#     → Need to handle with SMOTE or class_weight
#
# ⚠️  ZEROS
#     Column has many zero values
#     → Could be genuine zeros or missing data coded as 0
#     → Investigate before assuming
#
# ⚠️  MISSING
#     Column has significant missing values
#     → Decide: drop column, fill with mean/median/mode?
# ============================================================


# ============================================================
# PANDAS PROFILING vs MANUAL EDA — WHEN TO USE WHAT?
# ============================================================
#
# ┌─────────────────────┬──────────────┬───────────────────┐
# │                     │ Manual EDA   │ Pandas Profiling  │
# ├─────────────────────┼──────────────┼───────────────────┤
# │ Time required       │ 1-2 hours    │ 20 seconds        │
# │ Understanding depth │ Very deep    │ Surface level      │
# │ Customization       │ Full control │ Limited            │
# │ Large datasets      │ Works fine   │ Use minimal=True   │
# │ Interview/project   │ Essential    │ Good supplement    │
# │ Finding ALL issues  │ Might miss   │ Catches everything │
# └─────────────────────┴──────────────┴───────────────────┘
#
# THE GOLDEN WORKFLOW:
# Step 1 → Do MANUAL EDA first (understand data deeply)
# Step 2 → Run PANDAS PROFILING as Final Audit
#          (ensure you didn't miss any outliers or
#           weird correlations)
#
# Pandas Profiling is your FINAL AUDIT.
# You do manual EDA to understand deeply,
# then run Profiling to make sure you missed nothing.
# ============================================================


# ============================================================
# SPEED COMPARISON
# ============================================================
#
# MANUAL EDA to get all of this would require:
# df.describe()              → 2 minutes
# sns.histplot() per column  → 10 minutes
# sns.heatmap(df.corr())     → 5 minutes
# Missing value charts       → 10 minutes
# Scatter plots (pairplot)   → 5 minutes
# Writing warnings manually  → Impossible!
# ─────────────────────────────────────────
# Total manual time          → ~30-120 minutes
#
# With Pandas Profiling:
# ProfileReport(df)          → 20-30 seconds ✅
#
# Speed: You get 2 HOURS of manual coding work
# done in 20 SECONDS. That's the magic wand! 🪄
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Pandas Profiling = ydata-profiling (officially renamed)
#    Always import: from ydata_profiling import ProfileReport
#
# 2. ONE line of code generates a COMPLETE EDA report:
#    prof = ProfileReport(df)
#    prof.to_file("report.html")
#
# 3. Report includes: Overview, Variable Insights,
#    Interactions, Correlations, Missing Values, Warnings
#
# 4. THE WARNINGS SECTION is the most useful for ML:
#    → High Correlation → drop one of the features
#    → High Cardinality → drop or engineer the column
#    → Imbalanced → handle with SMOTE/class_weight
#    → Missing → decide drop or fill
#
# 5. Large datasets → use minimal=True to avoid freezing
#    prof = ProfileReport(df, minimal=True)
#
# 6. Three correlation types in the report:
#    Pearson (linear) → Spearman (monotonic) → Kendall (ordinal)
#    For most projects → focus on Pearson first
#
# 7. GOLDEN WORKFLOW:
#    Manual EDA first (deep understanding) →
#    then Pandas Profiling (final audit, catch what you missed)
#    Never skip manual EDA and rely only on profiling!
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Feature Engineering
#    Using the insights gained from EDA and Profiling
#    to CREATE new meaningful features and SELECT
#    the most useful ones for your ML model.
#    The warnings from Profiling directly guide
#    which features to drop or transform.
# ============================================================
from sklearn.model_selection import  cross_val_score
cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()