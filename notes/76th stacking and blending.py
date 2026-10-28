# ============================================================
# STACKING AND BLENDING
# ============================================================

# ============================================================
# WHAT IS IT?
#
# STACKING IS JUST LIKE VOTING, LIKE ASSIGNING WEIGHTS TO
# DIFFERENT MODELS, BUT IN CASE OF STACKING THE WEIGHTS ARE
# NOT ASSIGNED BY US. INSTEAD A NEW MODEL CALLED META MODEL
# OR LEVEL 1 MODEL IS TRAINED TO LEARN THE BEST WAY TO
# COMBINE THE PREDICTIONS OF THE BASE MODELS.
#
# While Voting with Weights usually means you (the human)
# decide the weights (e.g. "I trust the Random Forest 70%
# and the KNN 30%"), Stacking allows the Meta-Model to
# decide the weights AUTOMATICALLY based on the data.
#
# ITS LIKE GIVING THE OUTPUT DATA OF THE BASE MODELS TO
# ANOTHER MODEL TO LEARN HOW TO COMBINE THEM IN THE BEST
# WAY POSSIBLE INSTEAD OF JUST AVERAGING THEM LIKE IN
# VOTING.
# ============================================================


# ============================================================
# 1. DYNAMIC WEIGHTING — THE SECRET SAUCE
#
# In a simple weighted vote, the weights are FIXED.
# In Stacking, the weights can be CONDITIONAL.
#
# The Meta-Model (The Judge) might learn:
# → "When the patient is YOUNG, trust Random Forest more"
# → "When blood pressure is HIGH, trust Linear Model more"
# → "When models DISAGREE a lot, KNN is usually wrong"
#
# VISUAL — Voting vs Stacking:
#
# VOTING (fixed weights, decided by human):
# RF pred: 100  ──×0.7──┐
# KNN pred: 120 ──×0.3──┴──→ 0.7×100 + 0.3×120 = 106
# (same weights always, regardless of the input)
#
# STACKING (dynamic weights, learned by meta-model):
# RF pred: 100  ──┐
# KNN pred: 120 ──┴──→ Meta-Model ──→ 105
#                        ↑
#              learned from data that when RF says 100
#              and KNN says 120, answer is usually 105
#              AND this weight changes based on patient age!
# ============================================================


# ============================================================
# THE LEAKING PROBLEM
#
# THERE IS A PROBLEM CALLED "LEAKING" IN STACKING, WHERE
# THE META-MODEL CAN CHEAT BY SEEING THE TRUE TARGET
# VALUES DURING TRAINING AS WE USE THE WHOLE DATA TO
# PREDICT.
#                                                          
# TO AVOID THIS, STACKING USES CROSS-VALIDATION TO
# GENERATE PREDICTIONS FOR THE META-MODEL, ENSURING THAT
# IT ONLY SEES PREDICTIONS FROM MODELS THAT WERE TRAINED
# ON DIFFERENT FOLDS OF THE DATA.
#
# VISUAL — why leaking happens without CV:
#
# WITHOUT cross-validation (WRONG):
# Base models trained on ALL data
# → predict on SAME data they trained on
# → perfect predictions (memorized!)
# → Meta-model learns from "cheated" predictions ❌
#
# WITH cross-validation (CORRECT):
# Fold 1: base models trained on folds 2-5
#         → predict fold 1 (never seen) ✅
# Fold 2: base models trained on folds 1,3-5
#         → predict fold 2 (never seen) ✅
# Meta-model trained on these honest predictions ✅
# ============================================================


# ============================================================
# BLENDING (HOLD-OUT APPROACH)
#
# TO SOLVE THE LEAKING PROBLEM IN STACKING, THERE IS
# ANOTHER TECHNIQUE CALLED BLENDING (ALSO KNOWN AS
# HOLD-OUT), WHICH USES A HOLD-OUT SET TO TRAIN THE
# META-MODEL INSTEAD OF CROSS-VALIDATION. THIS WAY, THE
# META-MODEL NEVER SEES THE TRUE TARGET VALUES DURING
# TRAINING, BUT IT CAN BE LESS ROBUST IF YOUR DATASET IS
# SMALL.
#
# THE HOLD-OUT APPROACH (BLENDING):
# Think of it like a FINAL EXAM. If a teacher gives you
# the exact same questions during practice as the final
# exam, you aren't "smart" — you just have a good memory.
# The Hold-Out approach ensures the exam questions are
# brand new.
#
# THE CORE SPLIT:
# TRAINING SET (60%): Base models learn patterns here
# VALIDATION SET (20%): Base models predict here →
#                       meta-model trains on these predictions
# TEST SET (20%): Final verification — touched ONLY ONCE
#
# VISUAL — 3-way split for Blending:
#
# Full Dataset (100%)
# ├── Training Set (60%) → base models train here
# ├── Validation Set (20%) → base models predict here
# │                          meta-model trains here
# └── Test Set (20%) → final score, touched ONCE at end
# ============================================================


# ============================================================
# STACKING vs BLENDING
#
# ┌──────────────────┬──────────────────┬───────────────────┐
# │ Feature          │ Stacking         │ Blending          │
# ├──────────────────┼──────────────────┼───────────────────┤
# │ How meta-model   │ Cross-validation │ Hold-out          │
# │ gets its data    │ predictions      │ validation set    │
# │ Data efficiency  │ Uses ALL data    │ Wastes 20% on     │
# │                  │ (more robust)    │ validation        │
# │ Complexity       │ More complex     │ Simpler to code   │
# │ Best for         │ Small-medium     │ Large datasets    │
# │                  │ datasets         │ where 20% loss    │
# │                  │                  │ doesn't hurt      │
# └──────────────────┴──────────────────┴───────────────────┘
# ============================================================


# ============================================================
# FULL COMPARISON — ALL ENSEMBLE TYPES
#
# ┌─────────────┬────────────────┬─────────────────────────┐
# │ Technique   │ Logic          │ How it combines results  │
# ├─────────────┼────────────────┼─────────────────────────┤
# │ Voting      │ Equality       │ Simple Average /         │
# │             │                │ Majority Vote            │
# │ Stacking    │ Hierarchy      │ Meta-Model trained via   │
# │             │                │ Cross-Validation         │
# │ Blending    │ Hierarchy      │ Meta-Model trained on    │
# │             │                │ Hold-out set             │
# └─────────────┴────────────────┴─────────────────────────┘
# ============================================================


# ============================================================
# CODE — STACKING
# ============================================================

from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# 1. Define your "Base Experts" (Level 0)
base_models = [
    ('rf',    RandomForestRegressor(n_estimators=10,
                                    random_state=42)),
    ('knn',   KNeighborsRegressor(n_neighbors=5)),
    ('ridge', Ridge(alpha=1.0))
]

# 2. Define your "Final Judge" (Level 1)
# Linear Regression is the standard choice for Meta-Model
meta_model = LinearRegression()

# 3. Create the Stacking Ensemble
stacking_reg = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5  # 5-fold CV to generate honest meta-model inputs
)

# 4. Train
stacking_reg.fit(X_reg, y_reg)
print(f"Stacking Score: {stacking_reg.score(X_reg, y_reg):.4f}")


# ============================================================
# CODE — BLENDING (3-WAY SPLIT)
#
# THIS IS HOW U CAN PERFORM BLENDING BY SPLITTING INTO 3
# DATASETS: TRAINING, VALIDATION, AND TESTING
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# STEP 1: Split Data into 3 sets (60/20/20)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# STEP 2: Initialize and Train Base Models on 60%
model1 = XGBRegressor(n_estimators=100, learning_rate=0.1)
model2 = CatBoostRegressor(iterations=100, verbose=0)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

# STEP 3: Create "Blend" — base models predict on Val set
pred1 = model1.predict(X_val)
pred2 = model2.predict(X_val)

# Combine into a new DataFrame (Meta-Features)
# This is a table with 2 columns: XGB_Pred and Cat_Pred
blend_train = pd.DataFrame({'XGB': pred1, 'Cat': pred2})
# Meta-Model ONLY sees predictions, not original features!

# STEP 4: Train Meta-Model on blend predictions
meta_model = LinearRegression()
meta_model.fit(blend_train, y_val)

# STEP 5: Final Prediction on Test Set
test_pred1 = model1.predict(X_test)
test_pred2 = model2.predict(X_test)
blend_test = pd.DataFrame({'XGB': test_pred1,
                            'Cat': test_pred2})

final_prediction = meta_model.predict(blend_test)
print(f"Final MSE: {mean_squared_error(y_test, final_prediction):.4f}")


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Stacking = meta-model LEARNS how to combine base model
#    predictions — weights are dynamic, not fixed by human
#
# 2. LEAKING PROBLEM: if base models train and predict on
#    the same data, meta-model cheats → CV solves this
#
# 3. Stacking uses CV → more robust, uses all data
#    Blending uses hold-out → simpler, needs larger dataset
#
# 4. Meta-model only sees PREDICTIONS from base models,
#    not the original features
#
# 5. Meta-model is usually simple (Linear Regression,
#    Lasso) — the complexity is already in base models
#
# 6. 3-way split for blending: 60% train, 20% validation
#    (meta-model training), 20% test (touch ONCE at end)
#
# 7. INTERVIEW TRAP:
#    Q: "What is the difference between Stacking and
#        Blending?"
#    Right answer: "Both train a meta-model on base model
#    predictions. The difference is HOW they generate those
#    predictions. Stacking uses cross-validation — base
#    models predict on folds they never trained on, so the
#    meta-model gets honest predictions using ALL the data.
#    Blending uses a hold-out validation set — simpler but
#    wastes 20% of data on validation. Stacking is more
#    robust for smaller datasets, Blending is faster and
#    simpler for large ones."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → XGBoost
#    The optimized, faster version of Gradient Boosting
#    that dominates ML competitions worldwide.
# ============================================================