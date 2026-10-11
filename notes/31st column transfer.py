# ============================================================
# COLUMN TRANSFORMER — ONE PIPELINE FOR ALL TRANSFORMATIONS
# ============================================================

# ============================================================
# WHAT IS IT?
# If we need to perform DIFFERENT encoding methods on
# DIFFERENT columns it's not possible simply — we have to
# combine those encoded columns into the original dataset.
#
# That MANUAL process is:
# → Messy and hard to read
# → Prone to errors
# → Painful to repeat for test data
#
# ColumnTransformer AUTOMATES this by letting you apply
# different transformations to different columns in
# ONE SINGLE BLOCK OF CODE.
#
# ANALOGY:
# Imagine you're a chef preparing different ingredients:
# → Vegetables need to be chopped
# → Meat needs to be marinated
# → Spices need to be ground
#
# WITHOUT ColumnTransformer = doing each separately,
# then manually mixing them back together at the end.
#
# WITH ColumnTransformer = one master recipe that handles
# ALL ingredients simultaneously in one go! 🍳
# ============================================================


# ============================================================
# WHY DO WE NEED IT?
# ============================================================
#
# Imagine your CARS DATASET with these columns:
#
# brand     → Needs One Hot Encoding (with a threshold)
# fuel      → Needs One Hot Encoding
# owner     → Needs Ordinal Encoding (First > Second > Third)
# km_driven → Needs Standardization (StandardScaler)
#
# WITHOUT ColumnTransformer you'd have to:
# Step 1 → OneHotEncode fuel → get array1
# Step 2 → OrdinalEncode owner → get array2
# Step 3 → StandardScale km_driven → get array3
# Step 4 → OneHotEncode brand → get array4
# Step 5 → np.concatenate([array1, array2, array3, array4])
# Step 6 → Repeat ALL of this for test data!
# Step 7 → Pray you didn't mess up the column order 🙏
#
# WITH ColumnTransformer:
# Step 1 → Define the plan once
# Step 2 → transformer.fit_transform(X_train) ← done! ✅
# Step 3 → transformer.transform(X_test)      ← done! ✅
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   OrdinalEncoder)
from sklearn.model_selection import train_test_split


# ============================================================
# KEY PARAMETERS TO REMEMBER
# ============================================================
#
# transformers LIST:
# → A list of TUPLES, each tuple has THREE things:
#   1. Name       → anything you want (for identification)
#   2. Transformer → the encoder/scaler object
#   3. Columns    → list of column names or indices
#
# SYNTAX:
# [('name', transformer_object, [column_names])]
#
# remainder PARAMETER:
# → remainder='drop'        → DEFAULT
#                              DELETES columns you didn't mention
# → remainder='passthrough' → KEEPS unmentioned columns as they are
#
# WHICH TO USE?
# remainder='drop'        → when unmentioned cols are useless
# remainder='passthrough' → when unmentioned cols go into model
#                           as-is (like selling_price/target)
#
# SPECIAL OneHotEncoder PARAMETERS:
# sparse_output=False    → returns normal array not sparse matrix
# drop='first'           → avoids multicollinearity
# min_frequency=5        → groups rare categories into "infrequent"
#                          any category appearing < 5 times = grouped
# handle_unknown='infrequent_if_exist'
#                        → handles new unseen categories in test data
#                          gracefully instead of crashing
# ============================================================


# ============================================================
# CREATING THE COLUMN TRANSFORMER
# ============================================================

transformer = ColumnTransformer(transformers=[

    # TUPLE FORMAT: ('name', transformer_object, [columns])

    # fuel → One Hot Encoding
    # drop='first' → avoid multicollinearity
    # min_frequency=5 → group rare fuel types together
    # handle_unknown="infrequent_if_exist" → handle new fuel types
    ('tnf1', OneHotEncoder(sparse_output=False,
                           drop='first',
                           min_frequency=5,
                           handle_unknown='infrequent_if_exist'),
     ['fuel']),

    # owner → Ordinal Encoding
    # Manual order defined: First > Second > Third > Fourth
    # MUST pass list of lists for categories!
    ('tnf2', OrdinalEncoder(categories=[
                ['First Owner',
                 'Second Owner',
                 'Third Owner',
                 'Fourth Owner & Above']]),
     ['owner']),

    # km_driven → StandardScaler
    # Distance based models need this scaled
    # km_driven range is huge (1000 to 500000+)
    ('tnf3', StandardScaler(),
     ['km_driven']),

    # brand → One Hot Encoding with higher threshold
    # min_frequency=100 → only common brands get own column
    # rare brands → grouped into "infrequent" category
    # handle_unknown → safely handles new brands in test data
    ('tnf4', OneHotEncoder(min_frequency=100,
                           handle_unknown='infrequent_if_exist'),
     ['brand'])

],
# 'passthrough' keeps columns we didn't transform
# like selling_price or any other untouched columns
remainder='passthrough')


# ============================================================
# THE CORRECT ORDER — ALWAYS SPLIT FIRST THEN TRANSFORM
# ============================================================
#
# GOLDEN RULE: FIRST ALWAYS TRAIN TEST SPLIT THE DATA
# THEN PERFORM ANY TRANSFORM
#
# WHY? Same reason as StandardScaler:
# If you transform BEFORE splitting → data leakage!
# The transformer would learn mean/categories from test data too
# → model cheats → falsely good results → real world failure
#
# CORRECT ORDER:
# ┌─────────────────────────────────────────────────────┐
# │                                                     │
# │  Step 1 → train_test_split() first                  │
# │  Step 2 → transformer.fit_transform(X_train)        │
# │           (learns from train, transforms train)     │
# │  Step 3 → transformer.transform(X_test)             │
# │           (uses same rules, transforms test)        │
# │                                                     │
# │  ❌ NEVER transformer.fit_transform(X_test)         |
# │                                                     │
# └─────────────────────────────────────────────────────┘

# STEP 1 — Split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 2 — Fit and Transform the WHOLE table at once
# transformer learns all encoding rules from X_train only
X_train_transformed = transformer.fit_transform(X_train)

# STEP 3 — Transform test using SAME rules
# No fitting here → uses what it learned from train
X_test_transformed  = transformer.transform(X_test)

print("✅ ColumnTransformer applied successfully!")
print(f"X_train shape before : {X_train.shape}")
print(f"X_train shape after  : {X_train_transformed.shape}")
print(f"X_test shape before  : {X_test.shape}")
print(f"X_test shape after   : {X_test_transformed.shape}")
# NOTICE: columns INCREASE after transformation
# because One Hot Encoding creates multiple new columns!


# ============================================================
# VISUAL MENTAL MODEL — WHAT HAPPENS INSIDE
# ============================================================
#
# INPUT X_train (4 columns):
# ┌──────────┬──────┬────────┬──────────┐
# │ brand    │ fuel │ owner  │ km_driven│
# ├──────────┼──────┼────────┼──────────┤
# │ Maruti   │ CNG  │ First  │ 45000    │
# │ Honda    │ Petrol│Second │ 120000   │
# └──────────┴──────┴────────┴──────────┘
#
# AFTER ColumnTransformer:
# ┌──────────────────┬────────────┬───────┬────────────────┐
# │ brand_OHE cols   │ fuel_OHE   │ owner │ km_driven_scaled│
# │ (many columns)   │ (few cols) │ (0/1/2│ (-1.2, 0.3...) │
# ├──────────────────┼────────────┼───────┼────────────────┤
# │ 1 0 0 0 0 ...    │ 1 0        │  0    │    -0.82       │
# │ 0 1 0 0 0 ...    │ 0 1        │  1    │     1.23       │
# └──────────────────┴────────────┴───────┴────────────────┘
#
# All in ONE step! ✅
# ============================================================


# ============================================================
# COMMON MISTAKES TO AVOID
# ============================================================
#
# ❌ MISTAKE 1: Fitting on test data
#    transformer.fit_transform(X_test) → DATA LEAKAGE!
#    Always .transform() only on test data
#
# ❌ MISTAKE 2: Forgetting remainder='passthrough'
#    Default is 'drop' → target column gets deleted!
#    Always check what happens to unmentioned columns
#
# ❌ MISTAKE 3: Wrong categories format in OrdinalEncoder
#    categories=['First','Second'] → WRONG ❌
#    categories=[['First','Second']] → CORRECT ✅
#    Always list of lists!
#
# ❌ MISTAKE 4: sparse_output missing in OneHotEncoder
#    Without it → returns sparse matrix → causes errors
#    Always add sparse_output=False for normal array output
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. ColumnTransformer applies DIFFERENT transformations
#    to DIFFERENT columns in ONE single step
#    No more manual stitching of arrays!
#
# 2. Each transformer is a TUPLE of three things:
#    ('name', transformer_object, [column_names])
#
# 3. remainder parameter:
#    'drop'        → delete unmentioned columns (default)
#    'passthrough' → keep unmentioned columns as-is
#
# 4. ALWAYS split FIRST then transform:
#    fit_transform(X_train) → transform(X_test)
#    NEVER fit_transform on test data = data leakage ❌
#
# 5. min_frequency in OneHotEncoder:
#    Groups rare categories into "infrequent" bucket
#    Prevents model from learning from noise categories
#
# 6. handle_unknown='infrequent_if_exist':
#    Safely handles new unseen categories in test data
#    Without it → model CRASHES on new categories ❌
#
# 7. After transformation → number of columns INCREASES
#    because One Hot Encoding creates multiple new columns
#    from a single categorical column
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Sklearn Pipelines
#    Combining ColumnTransformer AND the ML model itself
#    into one single Pipeline object.
#    So preprocessing + training happens in one clean step
#    and you never accidentally forget to transform test data!
# ============================================================