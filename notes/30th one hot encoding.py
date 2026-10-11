# ============================================================
# ONE HOT ENCODING (OHE) — NOMINAL DATA
# ============================================================

# ============================================================
# WHAT IS IT?
#
# OHE IS USED FOR NOMINAL TYPE OF DATA WHERE THERE ARE NO
# SUCH THING AS RANKS.
#
# OHE creates a new binary column (0 or 1) for every unique
# category in your feature.
#
# Using the Customer Dataset — Gender column:
# Female becomes [1, 0]
# Male   becomes [0, 1]
# ============================================================


# ============================================================
# THE DUMMY VARIABLE TRAP — MULTICOLLINEARITY
#
# If you have n categories, you only need n-1 columns.
#
# Example — Gender:
# If gender_Male = 1 → it's Male
# If gender_Male = 0 → it MUST be Female
# The Female column carries ZERO new information.
#
# MULTICOLLINEARITY MEANS THAT THE INPUT COLUMNS HAVE A
# RELATIONSHIP WITH EACH OTHER. LIKE IN THIS CASE THE SUM
# WILL ALWAYS BE 1. SO THE MODEL WILL BE CONFUSED.
# SO GENERALLY WE DROP THE FIRST COLUMN.
#
# VISUAL:
#
# WITH both columns (trap):      WITH drop='first' (correct):
# ┌──────────────┬────────────┐  ┌──────────────┐
# │ Gender_Female│ Gender_Male│  │ Gender_Male  │
# ├──────────────┼────────────┤  ├──────────────┤
# │      1       │     0      │  │      0       │
# │      0       │     1      │  │      1       │
# │      1       │     0      │  │      0       │
# └──────────────┴────────────┘  └──────────────┘
# sum always = 1 ❌               0 means Female,
#                                 1 means Male ✅
# ============================================================


# ============================================================
# HIGH CARDINALITY PROBLEM
#
# If a column has too many categories (High Cardinality),
# like a "City" column with 100 cities, OHE will add 100
# columns making your data SPARSE and slow.
#
# WE CAN SEND THE LESS FREQUENT CATEGORIES INTO A COLUMN
# CALLED OTHER.
#
# Solution: Set a threshold. Keep the top categories and
# group the rest into "Other" before applying OHE.
#
# VISUAL:
#
# Brand counts:          After threshold (100):
# Maruti  → 450          Maruti  → own column
# Hyundai → 320          Hyundai → own column
# Honda   → 95     →     Honda   → "uncommon"
# Isuzu   → 12           Isuzu   → "uncommon"
# Renault → 8            Renault → "uncommon"
#
# sklearn handles this with min_frequency parameter:
# ohe = OneHotEncoder(min_frequency=0.1,
#                     handle_unknown='infrequent_if_exist')
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# SAMPLE DATA
# ============================================================

data = {
    'brand'        : ['Maruti', 'Skoda', 'Honda', 'Hyundai',
                      'Toyota', 'Ford', 'Renault', 'Hyundai',
                      'Maruti', 'Isuzu'],
    'km_driven'    : [145500, 120000, 140000, 127000, 120000,
                      45000, 175000, 5000, 90000, 40000],
    'fuel'         : ['Diesel', 'Diesel', 'Petrol', 'Diesel',
                      'Diesel', 'Diesel', 'LPG', 'Petrol',
                      'Diesel', 'Diesel'],
    'owner'        : ['First Owner', 'Second Owner', 'Third Owner',
                      'First Owner', 'First Owner', 'First Owner',
                      'Second Owner', 'First Owner', 'First Owner',
                      'First Owner'],
    'selling_price': [450000, 370000, 158000, 225000, 130000,
                      440000, 130000, 550000, 250000, 600000]
}

df = pd.DataFrame(data)
print(df.head())


# ============================================================
# WAY 1 — pandas get_dummies (NOT recommended for ML)
# ============================================================

# THE FIRST THING YOU CAN DO IS THE PANDAS GET_DUMMIES FUNCTION
pd.get_dummies(df, columns=["owner"], drop_first=True)
# passing owner in columns will remove the OWNER column and
# create different columns. drop_first=True will drop
# OWNER_First Owner column.

# WE GENERALLY DON'T USE PANDAS GET_DUMMIES IN ML PROJECTS
# BECAUSE IT DOESN'T REMEMBER THE ORDER OF COLUMNS SET EARLIER.
# EACH TIME IT MAY CHANGE THE ORDER OF THE COLUMNS.
#
# WHY THIS IS DANGEROUS:
# Train:  [Male=0, Female=1]   ← get_dummies decided this order
# Test:   [Female=0, Male=1]   ← get_dummies decided differently
# Model reads column 1 thinking it's Male but gets Female ❌
# sklearn's OneHotEncoder remembers the order from fit() ✅


# ============================================================
# WAY 2 — sklearn OneHotEncoder (recommended)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 0:4],   # all rows, columns 0 to 3
    df["selling_price"],
    test_size=0.2
)

# sparse=False skips the .toarray() step
# dtype=np.int32 saves memory vs default float
# drop='first' handles dummy variable trap
encoder = OneHotEncoder(drop="first", dtype=np.int32, sparse=False)

# fit_transform on train, transform only on test — golden rule
X_train_new = encoder.fit_transform(
    X_train[["brand", "fuel"]]
).toarray()
# it's a sparse matrix so we use .toarray() to convert it
# sparse matrix skips storing 0s to save space
# toarray() brings all those 0s back

X_test_new = encoder.transform(X_test[["brand", "fuel"]]).toarray()


# ============================================================
# HANDLING HIGH CARDINALITY — MANUAL THRESHOLD
# ============================================================

# Find which brands appear less than threshold times
counts    = df["brand"].value_counts()
threshold = 100
rep       = counts[counts <= threshold].index
# rep gives the NAMES of rare categories (index = category name
# in a value_counts Series)

pd.get_dummies(df["brand"].replace(rep, "uncommon"))
# rare brands are now merged into one "uncommon" column


# ============================================================
# HANDLING HIGH CARDINALITY — sklearn min_frequency
# ============================================================

# min_frequency=100 → same as your threshold
# handle_unknown='infrequent_if_exist' → rare categories
# get grouped automatically
ohe = OneHotEncoder(
    min_frequency=100,
    handle_unknown='infrequent_if_exist',
    drop='first',
    sparse_output=False
)

X_brand = ohe.fit_transform(df[['brand']])


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. OHE is for NOMINAL data only — no order/rank.
#    For ordinal data use OrdinalEncoder with defined order.
#
# 2. DUMMY VARIABLE TRAP:
#    n categories → only need n-1 columns
#    Always use drop='first' to avoid multicollinearity
#
# 3. MULTICOLLINEARITY = input columns have a relationship
#    with each other. In OHE, columns always sum to 1.
#    This confuses Linear/Logistic Regression models.
#
# 4. get_dummies vs OneHotEncoder:
#    get_dummies → quick exploration, NOT for ML pipelines
#    OneHotEncoder → always use this, remembers column order
#
# 5. HIGH CARDINALITY fix:
#    Manual: value_counts + threshold + replace with "uncommon"
#    sklearn: min_frequency parameter does the same thing
#
# 6. sparse=False or .toarray() — same result, pick one.
#    sparse_output=False is the newer parameter name.
#
# 7. GOLDEN RULE holds — fit on train, transform both.
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Column Transformer
#    Applying different encoders/scalers to different columns
#    in one clean pipeline step.
# ============================================================
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder,standardScaler
ord=OrdinalEncoder(categories=["Fresher","Junior","Senior"])
ohe=OneHotEncoder(drop="first",dtype=np.int32,sparse=False)
df["experience_level"]=ord.fit_transform(df[["experience_level"]])
df["city"]=ohe.fit_transform(df[["city"]]).toarray()
X=df.drop(columns=["salary"])
x_train,x_test,y_train,y_test=train_test_split(X,df["salary"],test_size=0.2)
scaler=standardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

