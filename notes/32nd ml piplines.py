# ============================================================
# SKLEARN PIPELINES — THE COMPLETE ASSEMBLY LINE
# ============================================================

# ============================================================
# WHAT IS IT?
# PIPELINES JUST CHAINS TOGETHER THE SEQUENCE OF STEPS
# AUTOMATICALLY.
#
# IT'S VERY USEFUL IF U GIVE YOUR MODEL THE NEW DATA OR
# THE TEST DATA. WITHOUT PIPELINES YOU HAVE TO MANUALLY
# DO EVERYTHING AGAIN LIKE HANDLE THE MISSING VALUES,
# TRANSFORM THEM, ENCODE AND OTHERS. WITH PIPELINES THIS
# WOULD AUTOMATICALLY HAPPEN TO YOUR TEST DATA.
#
# THE MAIN BENEFITS:
# → You don't have to write the steps again while
#   predicting new values
# → Helps you avoid errors
# → You don't even have to remember the sequence of steps
# → If you change anything in training data you DON'T
#   have to change it manually in testing data too
#
# ANALOGY:
# ColumnTransformer = a machine that processes ingredients
# Pipeline = the ENTIRE assembly line that takes raw
#            ingredients at one end and spits out a
#            finished product at the other end! 🏭
#
# A Pipeline bundles TWO main things into ONE object:
# 1. Preprocessing (Scaling, Encoding, Imputation etc.)
# 2. Model Training (Logistic Regression, Random Forest etc.)
# ============================================================


# ============================================================
# WHY USE A PIPELINE?
# ============================================================
#
# WITHOUT a pipeline:
# new_data_scaled = transformer.transform(new_data)  # manual!
# prediction = model.predict(new_data_scaled)         # manual!
#
# WITH a pipeline:
# prediction = pipe.predict(new_data)  # ONE LINE! ✅
# (The pipe REMEMBERS it needs to run the transformer first!)
#
# THREE BIG BENEFITS:
#
# 1. NO DATA LEAKAGE
#    → "Learning" only happens on training set
#    → Pipeline enforces this automatically
#
# 2. CLEANER DEPLOYMENT
#    → Only need to export ONE file (the pipe object)
#    → Your website/app just calls pipe.predict()
#    → No need to ship transformer + model separately
#
# 3. HYPERPARAMETER TUNING
#    → Can optimize preprocessing AND model parameters
#    → At the same time using GridSearchCV!
# ============================================================


# ============================================================
# SIMPLE IMPUTER — QUICK REFERENCE
# ============================================================
#
# SimpleImputer is a UNIVARIATE IMPUTER.
# This means it looks at ONLY ONE COLUMN AT A TIME
# to guess what the missing value should be.
#
# STRATEGIES IT USES:
#
# 'mean'         → average of the column
#                  Use for: Numerical, NO outliers
#
# 'median'       → middle value of the column
#                  Use for: Numerical WITH outliers ✅
#                  Best for Age (some passengers very old)
#
# 'most_frequent'→ most common value (mode)
#                  Use for: Categorical data ✅
#                  Best for Embarked (can't take mean of city!)
#
# 'constant'     → replace with a fixed value you choose
#                  Use for: Cabin → fill with "Unknown"
#
# TITANIC MISSING VALUES:
# Age      → Numerical + outliers → use MEDIAN ✅
# Embarked → Categorical          → use MOST_FREQUENT ✅
# Cabin    → 77% missing          → DROP the column ✅

from sklearn.impute import SimpleImputer

# Numerical → median
imputer_age      = SimpleImputer(strategy='median')

# Categorical → most frequent
imputer_embarked = SimpleImputer(strategy='most_frequent')

# Fit and Transform (learning happens here — TRAIN ONLY!)
# X_train['Age']      = imputer_age.fit_transform(
#                       X_train[['Age']])
# X_train['Embarked'] = imputer_embarked.fit_transform(
#                       X_train[['Embarked']])


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn import set_config
import seaborn as sns


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

df = sns.load_dataset("titanic")
df = pd.DataFrame(df)

print("=" * 55)
print("COLUMNS:")
print("=" * 55)
print(df.columns)

print("\nMISSING VALUES:")
print(df.isnull().sum())

# Drop columns we don't want
# deck, who, adult_male, sibsp, parch, alone → not useful
df.drop(columns=["deck", "who", "adult_male",
                  "sibsp", "parch", "alone"],
        inplace=True)


# ============================================================
# WHY MULTIPLE TRANSFORMERS INSTEAD OF ONE GIANT ONE?
# ============================================================
#
# WE WILL NOT BE USING JUST A SINGLE GIANT COLUMN
# TRANSFORMER AS SOME TRANSFORMATIONS WILL BE INCLUDING
# SOME MORE COLUMNS OR CREATE SOME MORE COLUMNS.
# THE INDEX WE DO WOULD BE GENERALLY WRONG AS THE INDEX
# IS BASED ON THE DATA GIVEN IN THE SINGLE COLUMN
# TRANSFORMER BUT THAT WOULD NOT BE THE CASE WITH
# MULTIPLE TRANSFORMERS.
#
# BY CREATING SEPARATE TRANSFORMERS AND PUTTING THEM IN
# A PIPELINE, THE OUTPUT OF TRANSFORMER1 BECOMES THE
# INPUT FOR TRANSFORMER2.
# IT KEEPS THE LOGIC CLEAN BECAUSE EACH STEP HANDLES
# THE "NEW" VERSION OF THE DATA.
#
# PIPELINE FLOW:
# Raw Data
#    ↓
# transformer1 → Fill missing values (Age, Embarked)
#    ↓
# transformer2 → One Hot Encode (sex, embarked)
#    ↓
# transformer3 → Scale numerical columns (MinMaxScaler)
#    ↓
# transformer4 → Select best K features (SelectKBest)
#    ↓
# transformer5 → Train model (DecisionTreeClassifier)
#    ↓
# Prediction ✅
# ============================================================


# ============================================================
# SET CONFIG — KEEP COLUMN NAMES THROUGHOUT PIPELINE
# ============================================================
#
# IF U WANT TO USE NAMES AS THE COLUMN NAMES IN THE
# PIPELINES INSTEAD OF THE INDICES U HAVE TO USE set_config.
#
# This ensures every step in your pipeline outputs a
# DataFrame with column NAMES instead of just indices.
#
# verbose_feature_names_out=False →
# VERBOSE OUTPUT DOESN'T LET THE COLUMN TRANSFORMER
# CHANGE THE NAMES OF THE COLUMN AS GENERALLY IT RENAMES
# THE COLUMN WITH "remainder" AS THE PREFIX.
# Setting False keeps original column names clean!

set_config(transform_output="pandas")


# ============================================================
# STEP 1 — TRANSFORMER 1: FILL MISSING VALUES
# ============================================================

transformer1 = ColumnTransformer(
    transformers=[
        # Fill Age with median (numerical + has outliers)
        ("impute_age",
         SimpleImputer(strategy="median"),
         ["age"]),

        # Fill Embarked with most frequent (categorical)
        ("impute_embarked",
         SimpleImputer(strategy="most_frequent"),
         ["embarked"])
    ],
    remainder="passthrough",
    verbose_feature_names_out=False  # keep original col names
)


# ============================================================
# STEP 2 — TRANSFORMER 2: ONE HOT ENCODING
# ============================================================

transformer2 = ColumnTransformer(
    transformers=[
        # One Hot Encode sex and embarked
        # handle_unknown="ignore" → safely handles new values
        ("ohe_sex_embarked",
         OneHotEncoder(sparse_output=False,
                       handle_unknown="ignore"),
         ["sex", "embarked"])
    ],
    remainder="passthrough",
    verbose_feature_names_out=False
)


# ============================================================
# STEP 3 — TRANSFORMER 3: SCALING
# ============================================================

transformer3 = ColumnTransformer(
    transformers=[
        # MinMaxScaler → scales values to 0-1 range
        ("scaled",
         MinMaxScaler(),
         ["age", "fare"])  # scale numerical columns
    ],
    remainder="passthrough"
)


# ============================================================
# STEP 4 — TRANSFORMER 4: FEATURE SELECTION
# ============================================================
#
# SelectKBest → selects only the TOP K most useful features
# score_func=chi2 → uses Chi-Squared test to rank features
# k=5 → SELECT ONLY THE TOP 5 COLUMNS
#
# WHY?
# Too many features = slower training + overfitting risk
# SelectKBest automatically keeps only the most useful ones

transformer4 = SelectKBest(score_func=chi2, k=5)


# ============================================================
# STEP 5 — TRANSFORMER 5: THE MODEL
# ============================================================

transformer5 = DecisionTreeClassifier()


# ============================================================
# BUILDING THE PIPELINE — TWO WAYS
# ============================================================

# WAY 1 — Using Pipeline object (explicit names)
# Each step is a tuple: ('name', transformer)
# Names are important → used in GridSearchCV later!
pipe = Pipeline([
    ("trf1", transformer1),   # imputation
    ("trf2", transformer2),   # encoding
    ("trf3", transformer3),   # scaling
    ("trf4", transformer4),   # feature selection
    ("trf5", transformer5)    # model
])

# WAY 2 — Using make_pipeline (auto-names the steps)
# Simpler syntax → just pass the objects directly
# Auto names: decisiontreeclassifier, columntransformer etc.
pipe1 = make_pipeline(transformer1, transformer2,
                      transformer3, transformer4,
                      transformer5)

# WHICH TO USE?
# Pipeline      → when you need GridSearchCV (need step names)
# make_pipeline → when you just want quick clean code


# ============================================================
# TRAIN TEST SPLIT — ALWAYS BEFORE FITTING!
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 1:],      # all columns except survived
    df["survived"],      # target column
    test_size=0.2,
    random_state=42
)


# ============================================================
# FITTING AND PREDICTING
# ============================================================
#
# IF U R JUST CHANGING COLUMNS LIKE ENCODING/SCALING
# U CAN USE pipe.fit_transform
# BUT U R USING THE PIPELINES TO TRAIN THE MODEL AS WELL
# SO U ALSO HAVE TO USE pipe.fit SO THAT U CAN USE
# pipe.predict AFTERWARDS FOR THE MODEL.
# U ALSO HAVE TO TRANSFORM YOUR TEST AND TRAIN DATA.
#
# SINCE WE ARE TRAINING THE ALGO IN THIS AS WELL
# SO WE WILL USE fit()

pipe.fit(X_train, y_train)

# See all steps inside the pipeline
print("\nPIPELINE STEPS:")
print(pipe.named_steps)

# Predict → pipeline automatically:
# 1. Imputes missing values in X_test
# 2. One Hot Encodes X_test
# 3. Scales X_test
# 4. Selects top K features
# 5. Predicts using Decision Tree
# ALL IN ONE LINE! 🔥
y_pred = pipe.predict(X_test)

from sklearn.metrics import accuracy_score
print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")


# ============================================================
# CROSS VALIDATION WITH PIPELINE
# ============================================================
#
# CROSS VALIDATION MEANS YOUR DATA WILL BE SPLIT AND
# TESTED ON 5 DIFFERENT SETS OF DATA LIKE IN THE CASE
# OF TEST_TRAIN_SPLIT AND THE SCORING PARAMETER TELLS
# YOU ABOUT WHICH PARAMETER YOU SHOULD DO THAT CHECKING.
#
# Pipeline + CrossValidation = no data leakage guaranteed!
# Each fold fits the ENTIRE pipeline on train folds only.

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(pipe, X_train, y_train,
                             cv=5, scoring='accuracy')

print(f"\nCV Scores     : {cv_scores}")
print(f"Mean Accuracy : {cv_scores.mean():.4f}")
print(f"Std Deviation : {cv_scores.std():.4f}")


# ============================================================
# GRIDSEARCHCV WITH PIPELINE — THE DOUBLE UNDERSCORE RULE
# ============================================================
#
# THE TRICKIEST PART OF USING GridSearchCV WITH A PIPELINE
# IS HOW YOU NAME THE PARAMETERS. YOU MUST USE THE
# DOUBLE UNDERSCORE (__) SYNTAX TO "REACH INSIDE" THE
# PIPELINE.
#
# THE DOUBLE UNDERSCORE RULE:
# [step_name]__[parameter_name]
#
# EXAMPLES:
# 'trf5__max_depth'           → max_depth of DecisionTree
# 'trf1__impute_age__strategy'→ strategy of Age imputer
#
# WHY DOUBLE UNDERSCORE?
# Pipeline has steps → each step has parameters
# __ lets you "drill down" into the nested structure
# pipe → trf5 (DecisionTree) → max_depth

params = {
    'trf5__max_depth': [1, 2, 3, 4, 5, None]
    # similarly u can add other parameters as well:
    # 'trf5__min_samples_split': [2, 5, 10],
    # 'trf5__criterion': ['gini', 'entropy']
}

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)


# ============================================================
# INSPECTING GRIDSEARCHCV RESULTS
# ============================================================
#
# ONCE THE GRID SEARCH IS FINISHED YOU CAN ASK FOR
# THE "WINNING" SETTINGS:

# 1. The best accuracy achieved
print(f"\nBest Score  : {grid.best_score_:.4f}")

# 2. The exact parameters that gave that score
print(f"Best Params : {grid.best_params_}")

# 3. Use the best model immediately
best_model = grid.best_estimator_
y_pred     = best_model.predict(X_test)
print(f"Best Model Test Accuracy: "
      f"{accuracy_score(y_test, y_pred):.4f}")


# ============================================================
# EXPORTING THE PIPELINE — DEPLOYMENT READY!
# ============================================================
#
# ONE OF THE BIGGEST BENEFITS OF PIPELINE:
# You only need to export ONE file for deployment!
# The pipe object contains EVERYTHING:
# → All transformers (imputer, encoder, scaler)
# → The trained model
# → The correct order of operations
#
# Your Streamlit app just loads this one file and
# calls pipe.predict(raw_input) — done! 🚀

import pickle

# Save the pipeline
pickle.dump(pipe, open('pipe.pkl', 'wb'))
print("\n✅ Pipeline saved as pipe.pkl")
print("   Load it anywhere with:")
print("   pipe = pickle.load(open('pipe.pkl', 'rb'))")
print("   prediction = pipe.predict(raw_new_data)")


# ============================================================
# PIPELINE vs COLUMN TRANSFORMER — SUMMARY
# ============================================================
#
# ┌─────────────────────┬──────────────────────────────────┐
# │                     │                                  │
# │ ColumnTransformer   │ Handles FEATURES (X) only        │
# │                     │ Scales and encodes columns       │
# │                     │ Output = transformed X           │
# ├─────────────────────┼──────────────────────────────────┤
# │ Pipeline            │ Handles ENTIRE workflow          │
# │                     │ Includes ColumnTransformer +     │
# │                     │ the ML Model                     │
# │                     │ Output = predictions             │
# └─────────────────────┴──────────────────────────────────┘
#
# USE BOTH TOGETHER:
# ColumnTransformer → INSIDE the Pipeline as a step
# Pipeline → wraps EVERYTHING into one clean object
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Pipeline = chains ALL steps automatically
#    preprocessing + model = ONE single object
#
# 2. pipe.fit(X_train, y_train)
#    → fits ALL steps on training data only
#    pipe.predict(X_test)
#    → automatically transforms X_test through all steps
#    → then predicts → NO manual transformation needed!
#
# 3. TWO ways to build:
#    Pipeline([('name', step)]) → explicit names (use with GridSearchCV)
#    make_pipeline(step1, step2) → auto names (quick and clean)
#
# 4. DOUBLE UNDERSCORE RULE for GridSearchCV:
#    'step_name__parameter' → reach inside pipeline steps
#    'trf5__max_depth' → max_depth of step named trf5
#
# 5. set_config(transform_output="pandas")
#    → keeps column names throughout pipeline
#    verbose_feature_names_out=False
#    → prevents "remainder__" prefix on column names
#
# 6. DEPLOYMENT BENEFIT:
#    pickle.dump(pipe) → export ONE file
#    Contains everything → transformers + model
#    pipe.predict(raw_data) → works on completely raw input!
#
# 7. Pipeline + CrossValidation = zero data leakage
#    Each fold fits entire pipeline on train folds only
#    Most production-ready evaluation method
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Machine Learning Algorithms
#    Now that data is perfectly preprocessed via Pipeline,
#    we start training actual ML models — Linear Regression,
#    Logistic Regression, Decision Trees, Random Forest,
#    and more — comparing them inside the same Pipeline!
# ============================================================