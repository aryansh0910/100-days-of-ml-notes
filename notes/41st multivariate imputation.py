# ============================================================
# MULTIVARIATE IMPUTATION — KNN + ITERATIVE IMPUTER
# ============================================================

# ============================================================
# WHAT IS IT?
#
# Multivariate imputation is the "high-intelligence" way to
# fill missing data. Unlike Univariate imputation (which only
# looks at one column at a time), Multivariate imputation
# looks at the RELATIONSHIPS between all your variables to
# guess the missing value.
#
# Think of it as a "predictive" process. If a person's Age
# is missing, the algorithm doesn't just give them the
# average age of the whole dataset. Instead, it looks at
# their Salary, Job Title, and Education and says:
#
# "Based on this person's $80k salary and Senior Manager
#  title, they are likely around 45 years old."
#
# UNIVARIATE vs MULTIVARIATE — the core difference:
#
# Univariate (SimpleImputer):
# Age column → mean of Age = 34 → fill all NaNs with 34
# Ignores every other column ❌
#
# Multivariate (KNN / Iterative):
# Age NaN → look at Salary=80k, Title=Senior Manager
#         → similar rows have Age ~45
#         → fill with 45 ✅ much smarter!
# ============================================================


# ============================================================
# A. KNN IMPUTER — "THE NEIGHBOURHOOD WATCH"
#
# LOGIC:
# IT WOULD FILL THE VALUES BASED ON THE VALUES THAT ARE
# SIMILAR TO IT. THAT SIMILARITY IS BASED ON EUCLIDEAN
# DISTANCE. THE VALUE OF K MEANS HOW MANY NEAREST NEIGHBOURS
# TO CHECK.
#
# THIS TECHNIQUE IS MORE ACCURATE THAN JUST FILLING RANDOM
# VALUES.
#
# Step by step:
# 1. Find the K most similar rows to the row with NaN
# 2. Take the average of those K neighbours' values
# 3. Fill the NaN with that average
#
# VISUAL (K=2):
#
# Row with NaN:  Age=NaN, Salary=80k, Exp=10yrs
#
# All other rows:
# Row 3: Age=45,  Salary=78k, Exp=9yrs  ← distance: small ✅
# Row 7: Age=43,  Salary=82k, Exp=11yrs ← distance: small ✅
# Row 1: Age=25,  Salary=30k, Exp=2yrs  ← distance: large ❌
#
# KNN picks Row 3 and Row 7 (closest 2 neighbours)
# Fill value = (45 + 43) / 2 = 44 ✅
#
# DISADVANTAGES:
# → TIME TAKING as it calculates distance between every row
#   1 million rows = billions of distance calculations = slow
# → U HAVE TO DEPLOY YOUR WHOLE DATASET TO THE SERVER
#   so MORE MEMORY IS UTILISED at production time
# ============================================================

from sklearn.impute import KNNImputer

# weights='distance' → closer neighbours get MORE influence
# n_neighbors=5      → look at 5 most similar rows
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')

# In a ColumnTransformer:
# ('knn_impute', KNNImputer(n_neighbors=5, weights='distance'),
#  ['Age', 'Fare'])


# ============================================================
# B. ITERATIVE IMPUTER — "THE ROUND-ROBIN PREDICTOR"
#
# LOGIC:
# Inspired by MICE (Multivariate Imputation by Chained
# Equations) from R. It treats the column with missing values
# as a TARGET (y) and all other columns as FEATURES (X).
# It literally builds a mini regression model to predict
# the missing values!
#
# THE PROCESS — Round Robin:
#
# Step 1 — INITIAL GUESS:
# Fill all NaNs with a basic placeholder (usually mean)
#
# Step 2 — THE LOOP:
# ┌─────────────────────────────────────────────────────┐
# │ Pick Column A (Age)                                 │
# │ → Use Fare, Pclass as X → predict missing Age       │
# │                                                     │
# │ Pick Column B (Fare)                                │
# │ → Use updated Age, Pclass as X → predict missing   │
# │                                                     │
# │ Pick Column C (Pclass)                              │
# │ → Use updated Age, Fare as X → predict missing     │
# │                                                     │
# │ Repeat for max_iter cycles until values stabilise   │
# └─────────────────────────────────────────────────────┘
#
# WHY BETTER THAN KNN?
# KNN just averages neighbours — no understanding of WHY
# IterativeImputer learns the actual mathematical
# relationship between columns — much smarter!
#
# You can even CHOOSE which model does the predicting:
# LinearRegression → fast, works well for linear data
# RandomForest     → handles complex non-linear patterns
# ============================================================

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # Required!
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
# We can choose our "brain" — the model that predicts values

# 1. Create a dataset with a clear relationship (Age ≈ 10 * Class)
df = pd.DataFrame({
    'Class': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'Age'  : [10, 20, 30, 40, 50, np.nan, 22, np.nan, 38, 52]
})

# 2. Initialize the Smart Imputer
# estimator = the model used to predict missing values
# max_iter  = how many round-robin cycles to run
# verbose=2 → prints each iteration so you can watch it work
lr     = LinearRegression()
it_imp = IterativeImputer(estimator=lr, max_iter=10, verbose=2)

# 3. Fit and Transform
# fit step  → imputer LEARNS the relationship between columns
# transform → uses that relationship to fill NaNs
df_filled = it_imp.fit_transform(df)

# Convert back to DataFrame to see the smart values
print(pd.DataFrame(df_filled, columns=['Class', 'Age']))
# Row 5 (Class=1) → Age should be ~10 (learned from pattern)
# Row 7 (Class=3) → Age should be ~30 (learned from pattern)
# Compare with mean fill which would give both rows ~34 ❌


# ============================================================
# KNN vs ITERATIVE — QUICK COMPARISON
#
# ┌─────────────────┬──────────────────┬───────────────────┐
# │                 │ KNNImputer       │ IterativeImputer  │
# ├─────────────────┼──────────────────┼───────────────────┤
# │ How it fills    │ Average of K     │ Regression model  │
# │                 │ nearest rows     │ per column        │
# │ Speed           │ Slow (distances) │ Faster            │
# │ Memory          │ Needs full data  │ Needs full data   │
# │                 │ on server        │ on server         │
# │ Best for        │ Small-medium     │ Any size, esp.    │
# │                 │ datasets         │ with clear        │
# │                 │                  │ relationships     │
# │ Customisable    │ Only K and       │ Swap the model    │
# │                 │ weights          │ (LR, RF, etc.)    │
# └─────────────────┴──────────────────┴───────────────────┘
#
# WHEN TO USE WHICH:
# Clear relationship between columns → IterativeImputer
# No clear relationship, just similarity → KNNImputer
# Large dataset (1M+ rows) → avoid KNN, use Iterative
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Multivariate = uses OTHER columns to predict missing value
#    Univariate = uses only SAME column stats (mean/median)
#
# 2. KNNImputer:
#    → Finds K most similar rows using Euclidean distance
#    → Fills with average of those K neighbours
#    → Slow on large data, needs full dataset on server
#    → weights='distance' → closer neighbours matter more
#
# 3. IterativeImputer:
#    → Treats missing column as y, all others as X
#    → Builds a regression model to PREDICT the missing value
#    → Runs in rounds until values stabilise (max_iter)
#    → You choose the estimator — LR, RF, anything sklearn
#    → Needs enable_iterative_imputer import (still experimental)
#
# 4. INTERVIEW TRAP:
#    Q: "What is the difference between KNN imputer and
#        Iterative Imputer?"
#    Right answer: "KNN finds similar rows and averages their
#    values — no model learning involved. Iterative Imputer
#    builds an actual regression model per column and learns
#    the mathematical relationship between features to predict
#    the missing value. Iterative is smarter but both need
#    full data at deployment time."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Feature Engineering
#    Creating new meaningful columns from existing ones
#    to give your model more signal to learn from.
# ============================================================


from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer,SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
col=ColumnTransformer([("tg1",KNNImputer(n_neighbors=5, weights='distance'),['Age']),
                       ("tfg2",SimpleImputer(strategy="most frequent"),["city"]),
                       ("tfg2",StandardScaler(),["Salary"])
                       ],remainder='passthrough')