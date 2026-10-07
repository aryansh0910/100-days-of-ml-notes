# ============================================================
# FEATURE ENGINEERING — EXTRACTING GOLD FROM RAW DATA
# ============================================================

# ============================================================
# WHAT IS IT?
# FEATURE ENGINEERING IS KNOWN AS THE KNOWLEDGE TO EXTRACT
# THE USEFUL DATA THAT WE COULD GIVE TO OUR MODEL
# FROM THE RAW DATA.
#
# Raw data is like uncut gold — valuable but unusable.
# Feature Engineering is the REFINING process that turns
# raw data into pure gold your model can actually use.
#
# It is divided into FOUR parts:
# 1. Feature Transformation  → change/fix existing columns
# 2. Feature Construction    → create NEW columns
# 3. Feature Selection       → remove USELESS columns
# 4. Feature Extraction      → compress MANY columns into FEW
#
# ANALOGY:
# Think of your dataset as raw ingredients in a kitchen.
# Feature Engineering = the chef's preparation work:
# Cleaning, chopping, combining, and selecting ingredients
# BEFORE putting them in the oven (the model).
# Bad prep = bad dish, no matter how good the oven is!
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Load Titanic dataset
df = pd.read_csv("22nd_train_data.csv")

print("=" * 55)
print("ORIGINAL DATASET")
print("=" * 55)
print(df.head(3))
print(f"\nShape: {df.shape}")
print()


# ============================================================
# PART 1 — FEATURE TRANSFORMATION
# ============================================================
#
# This is about changing the MATH or FORMAT of a single
# column to make it easier for the model to understand.
# You are NOT creating new columns — just fixing existing ones.
#
# FOUR main transformations:
# A) Handling Missing Values (Imputation)
# B) Handling Categorical Data (Encoding)
# C) Outlier Handling
# D) Scaling
# ============================================================

print("=" * 55)
print("PART 1 — FEATURE TRANSFORMATION")
print("=" * 55)

# ── A) HANDLING MISSING VALUES (Imputation) ──────────────────
#
# FILLING IN THE NaN VALUES IN AGE WITH THE MEAN OR MEDIAN.
#
# DECISION RULES:
# U CAN REMOVE THEM AS WELL IF THEY ARE VERY LOW IN NUMBER
# OR U CAN JUST FILL THEM AS WELL.
# IF CATEGORY DATA THEN FILL WITH THE MOST FREQUENT CATEGORY.
#
# RULE OF THUMB:
# < 5% missing   → DROP the rows (safe to remove)
# 5-30% missing  → FILL with mean/median (numerical)
#                         or mode (categorical)
# > 30% missing  → DROP the entire column
#
# WHY MEDIAN OVER MEAN?
# Mean is affected by outliers.
# Example: Age column with one value of 200
# → Mean gets pulled up → bad fill value
# → Median stays realistic → better fill value

print("\nA) Missing Values BEFORE imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Numerical → fill with median
num_imputer = SimpleImputer(strategy='median')
df['Age'] = num_imputer.fit_transform(df[['Age']])

# Categorical → fill with most frequent (mode)
cat_imputer = SimpleImputer(strategy='most_frequent')
if 'Embarked' in df.columns:
    df['Embarked'] = cat_imputer.fit_transform(
                     df[['Embarked']]).ravel()

print("\nMissing Values AFTER imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("✅ No more missing values!")

# ── B) HANDLING CATEGORICAL DATA (Encoding) ──────────────────
#
# TURNING "Male/Female" INTO 0 AND 1.
# YOU HAVE TO DO ONE HOT ENCODING AS OUR MODEL WOULD
# ONLY UNDERSTAND NUMBERS — NOT WORDS.
#
# TWO TYPES OF ENCODING:
#
# Label Encoding → Male=0, Female=1
# → Use for ORDINAL data (has natural order)
# → Example: Low=0, Medium=1, High=2
# → NEVER use for nominal data — model thinks 2 > 1 > 0!
#
# One Hot Encoding → creates separate column per category
# → Use for NOMINAL data (no natural order)
# → Example: Male/Female, City names, Colors
# → pd.get_dummies() does this automatically

print("\nB) Encoding Categorical Data:")

# ONE HOT ENCODING using pd.get_dummies
if 'Sex' in df.columns:
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
    # drop_first=True → drops one column to avoid multicollinearity
    # Male=0, Female=1 now represented as one column "Sex_male"
    print("Sex column encoded using One Hot Encoding ✅")

print(df.head(2))


# ── C) OUTLIER HANDLING ───────────────────────────────────────
#
# USING THE BOX PLOT TO FIND "CRAZY" VALUES AND
# CAPPING THEM SO THEY DON'T TRICK THE MODEL.
#
# TWO APPROACHES:
# REMOVE → delete the outlier rows entirely
#           Use when outliers are clearly data ERRORS
#           Example: Age = -3, Age = 200
#
# CAP    → replace outlier with boundary value (Winsorizing)
#           Use when outliers are REAL but extreme
#           Example: A billionaire's salary in income data
#           Cap at 99th percentile → keeps the row, fixes value
#
# IQR METHOD to find boundaries:
# Q1 = 25th percentile
# Q3 = 75th percentile
# IQR = Q3 - Q1
# Lower = Q1 - 1.5 × IQR  → anything below = outlier
# Upper = Q3 + 1.5 × IQR  → anything above = outlier

print("\nC) Outlier Handling (Fare column):")

Q1  = df['Fare'].quantile(0.25)
Q3  = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
upper_cap = df['Fare'].quantile(0.99)  # cap at 99th percentile

print(f"Fare max BEFORE capping: {df['Fare'].max():.2f}")

# CAP the outliers at 99th percentile
df['Fare'] = df['Fare'].clip(upper=upper_cap)

print(f"Fare max AFTER capping : {df['Fare'].max():.2f}")
print("✅ Outliers capped at 99th percentile!")


# ── D) SCALING ────────────────────────────────────────────────
#
# AS SOME ALGORITHMS CALCULATE THE DISTANCE BETWEEN TWO
# POINTS SO IT WOULD CONSIDER VALUES LIKE 70000 AND 19
# VERY FAR APART. LIKE IF ONE IS THE SALARY AND OTHER IS
# THE AGE AND THEN IT WOULD ALSO CONSIDER THE AGE COLUMN
# LESS IMPORTANT SO WE HAVE TO SCALE ACCORDINGLY.
#
# IF AGE IS 0-80 AND FARE IS 0-500, THE MODEL MIGHT THINK
# FARE IS "MORE IMPORTANT" JUST BECAUSE THE NUMBERS
# ARE BIGGER. WE SCALE THEM TO BE BETWEEN 0 AND 1.
#
# TWO TYPES OF SCALING:
#
# StandardScaler (Z-score normalization):
# → Converts to mean=0, std=1
# → USE WHEN: data is normally distributed
# → Formula: (x - mean) / std
# → Does NOT bound to 0-1 range
#
# MinMaxScaler (Normalization):
# → Converts to range 0 to 1
# → USE WHEN: data has outliers OR you need 0-1 range
# → Formula: (x - min) / (max - min)
# → Sensitive to outliers
#
# WHICH ALGORITHMS NEED SCALING?
# ✅ NEED scaling  → KNN, SVM, Neural Networks,
#                    Logistic Regression, PCA
# ❌ DON'T need it → Random Forest, Decision Trees,
#                    XGBoost (tree based = not distance based)

print("\nD) Scaling Numerical Columns:")

cols_to_scale = ['Age', 'Fare']
scaler = StandardScaler()

df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print(f"Age  mean after scaling: {df['Age'].mean():.4f}  "
      f"(should be ~0)")
print(f"Age  std after scaling : {df['Age'].std():.4f}   "
      f"(should be ~1)")
print("✅ Scaling complete!")


# ============================================================
# PART 2 — FEATURE CONSTRUCTION
# ============================================================
#
# THIS IS THE PROCESS OF CREATING A NEW COLUMN FROM THE
# EXISTING DATA SO THAT IT MAY BECOME MORE USEFUL.
#
# This is the CREATIVE part of feature engineering.
# You look at existing columns and COMBINE them to create
# a new, more powerful column.
#
# ANALOGY:
# Like a chef who combines flour + eggs + sugar
# to create something new (a cake) that is more
# valuable than any ingredient alone!
#
# TITANIC EXAMPLE:
# SibSp (siblings/spouses) alone → not very informative
# Parch (parents/children) alone → not very informative
# BUT combined:
# FamilySize = SibSp + Parch + 1 (the passenger themselves)
# NOW the model knows: alone vs small family vs large family
# This single new feature can be MORE powerful than both
# original features combined!

print("\n" + "=" * 55)
print("PART 2 — FEATURE CONSTRUCTION")
print("=" * 55)

# Create FamilySize feature
if 'SibSp' in df.columns and 'Parch' in df.columns:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # Create IsAlone feature from FamilySize
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    # 1 = travelling alone, 0 = travelling with family

    print("New features created:")
    print(df[['SibSp', 'Parch', 'FamilySize', 'IsAlone']].head(5))
    print("\n✅ FamilySize and IsAlone features constructed!")

# MORE EXAMPLES OF FEATURE CONSTRUCTION:
# ─────────────────────────────────────────────────────────────
# Car dataset:
# Car Age = Current Year - Manufacturing Year
# → More intuitive than raw year for the model
#
# E-commerce dataset:
# Revenue per Visit = Total Revenue / Number of Visits
# → Better signal than either column alone
#
# Text dataset (Resume Screener!):
# Keyword Match % = matched_keywords / total_keywords × 100
# → More meaningful than raw keyword count


# ============================================================
# PART 3 — FEATURE SELECTION
# ============================================================
#
# LIKE IN MOST OF THE DATASETS CONTAINING IMAGE PIXEL VALUES
# MOST OF THE COLUMNS ARE JUST ZERO SO WE CAN JUST REMOVE
# THEM OTHERWISE OUR MACHINE WILL TAKE A HUGE AMOUNT OF TIME.
#
# You don't want to give your model EVERY column.
# Columns like PassengerId, Name, Ticket Number are usually
# just "noise" — they don't help predict survival.
#
# TOO MANY FEATURES = CURSE OF DIMENSIONALITY
# Model gets confused by irrelevant noise columns
# Training becomes slower unnecessarily
# Overfitting risk increases
#
# THREE METHODS TO SELECT FEATURES:
# 1. Correlation Heatmap  → drop low correlation with target
# 2. Feature Importance   → tree models tell you importance
# 3. Domain Knowledge     → use your brain! PassengerId = useless

print("\n" + "=" * 55)
print("PART 3 — FEATURE SELECTION")
print("=" * 55)

# METHOD 1 → Correlation with target variable
if 'Survived' in df.columns:
    numerical_df = df.select_dtypes(include=[np.number])
    correlation_with_target = numerical_df.corr()['Survived']\
                              .abs().sort_values(ascending=False)
    print("Correlation with Survived (target):")
    print(correlation_with_target)
    print()

    # Drop features with very low correlation (< 0.05)
    weak_features = correlation_with_target[
                    correlation_with_target < 0.05].index.tolist()
    weak_features = [f for f in weak_features if f != 'Survived']
    print(f"Weak features to consider dropping: {weak_features}")

# METHOD 2 → Drop obviously useless columns (domain knowledge)
cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=cols_to_drop)
print(f"\nDropped noise columns: {cols_to_drop}")
print(f"Shape after selection: {df.shape}")
print("✅ Feature selection complete!")
+

# ============================================================
# PART 4 — FEATURE EXTRACTION
# ============================================================
#
# THIS IS ADVANCED. IT'S USED WHEN YOU HAVE TOO MANY
# COLUMNS (LIKE 100+) AND YOU WANT TO SHRINK THEM DOWN
# INTO 5 "SUPER COLUMNS" THAT CONTAIN ALL THE IMPORTANT INFO.
#
# PCA (Principal Component Analysis) is the most famous tool.
# IT'S LIKE "ZIPPING" A FOLDER OF DATA.
#
# HOW PCA WORKS:
# → Finds the DIRECTIONS of maximum variance in your data
# → Projects all your features onto these new directions
# → Each new direction = one "Principal Component"
# → You keep only the top N components
# → Result: fewer columns, most information preserved
#
# EXAMPLE:
# 100 pixel features → PCA → 5 super components
# Those 5 components capture 95% of the information!
# Training is now 20x faster with barely any accuracy loss.
#
# WHEN TO USE PCA:
# ✅ Dataset has 50+ features
# ✅ Many features are correlated with each other
# ✅ Model is training too slowly
# ✅ Image/text data with huge feature spaces
#
# WHEN NOT TO USE PCA:
# ❌ You need to explain which features matter (PCA loses names)
# ❌ Dataset has only 10-15 features (not worth it)

print("\n" + "=" * 55)
print("PART 4 — FEATURE EXTRACTION (PCA)")
print("=" * 55)

# Demo PCA on numerical columns
numerical_cols = df.select_dtypes(include=[np.number])\
                   .drop(columns=['Survived'], errors='ignore')

print(f"Features BEFORE PCA: {numerical_cols.shape[1]} columns")
print(f"Columns: {list(numerical_cols.columns)}")

# Apply PCA → compress to 3 super components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(numerical_cols.fillna(0))

print(f"\nFeatures AFTER PCA : {X_pca.shape[1]} columns")
print(f"Variance explained : "
      f"{pca.explained_variance_ratio_.sum()*100:.1f}%")
print(f"Per component      : "
      f"{[f'{v*100:.1f}%' for v in pca.explained_variance_ratio_]}")
print("✅ PCA complete — data compressed!")

# HOW TO READ explained_variance_ratio_:
# [0.45, 0.30, 0.15] means:
# Component 1 → captures 45% of all information
# Component 2 → captures 30% of all information
# Component 3 → captures 15% of all information
# Total       → 90% of information in just 3 columns!


# ============================================================
# COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================
#
# In a real project, you combine ALL four parts:
#
# RAW DATA
#    ↓
# TRANSFORMATION  → fix missing, encode, remove outliers, scale
#    ↓
# CONSTRUCTION    → create new meaningful columns
#    ↓
# SELECTION       → remove noise and useless columns
#    ↓
# EXTRACTION      → compress if still too many features
#    ↓
# CLEAN DATA → ready to feed into your ML model! ✅
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Feature Engineering = 4 parts:
#    Transformation → Construction → Selection → Extraction
#
# 2. TRANSFORMATION fixes existing columns:
#    Missing values → median (numerical), mode (categorical)
#    Encoding       → One Hot for nominal, Label for ordinal
#    Outliers       → remove errors, cap real extreme values
#    Scaling        → StandardScaler for normal distribution
#                     MinMaxScaler for bounded 0-1 range
#
# 3. CONSTRUCTION creates new columns:
#    FamilySize = SibSp + Parch + 1 (Titanic example)
#    Car Age = Current Year - Manufacturing Year
#    Combine weak features into one strong feature
#
# 4. SELECTION removes useless columns:
#    Use correlation heatmap → drop low correlation features
#    Use domain knowledge → drop ID, Name, Ticket columns
#    Too many features = slower training + overfitting risk
#
# 5. EXTRACTION compresses many features into few:
#    PCA = "zip" 100 features into 5 super components
#    Use when 50+ features or model training too slow
#    Downside: lose feature names and interpretability
#
# 6. Algorithms that NEED scaling:
#    KNN, SVM, Neural Networks, Logistic Regression, PCA
#    Tree based models (Random Forest, XGBoost) do NOT need it
#
# 7. Feature Engineering is the SECRET SAUCE:
#    Same algorithm + better features = much better model
#    This is what separates good ML engineers from average ones
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Model Training & Evaluation
#    Now that data is clean and features are engineered,
#    we finally train multiple ML algorithms, compare them,
#    and use hyperparameter tuning to squeeze out the
#    best possible performance from our chosen model.
# ============================================================