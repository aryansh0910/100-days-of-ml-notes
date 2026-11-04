# ============================================================
# XGBOOST CLASSIFIER
# ============================================================

# ============================================================
# WHAT IS IT?
#
# While XGBoost Regression was about predicting a specific
# NUMBER (like a blood sugar level), XGBoost Classifier is
# about predicting a PROBABILITY (like "Does this patient
# have Diabetes?").
#
# It is the most popular tool for classification because it
# handles messy data, missing values, and non-linear
# relationships better than almost any other algorithm.
# ============================================================


# ============================================================
# 1. THE CORE LOGIC — LOG-ODDS AND PROBABILITIES
#
# XGBoost Classification doesn't just "guess" a category.
# It builds trees based on LOG-ODDS.
#
# STEP 1 — BASE GUESS:
# Starts with a default probability for everyone
# (usually 0.5 or 50%)
#
# STEP 2 — CALCULATE RESIDUALS:
# Looks at the difference between the ACTUAL LABEL
# (1 for Yes, 0 for No) and the current probability
#
# STEP 3 — BUILD A TREE:
# Builds a tree to predict these residuals
#
# STEP 4 — TRANSFORMATION:
# Uses a SIGMOID FUNCTION to turn those tree outputs
# back into a probability between 0 and 1
#
# STEP 5 — THE THRESHOLD:
# If final probability > 0.5 → predict "Class 1" (Positive)
#
# VISUAL — how probabilities update over trees:
#
# True label = 1 (Diabetic)
#
# Start: P(Diabetic) = 0.50  → Residual = 1 - 0.50 = 0.50
# Tree 1 corrects:   P = 0.65 → Residual = 1 - 0.65 = 0.35
# Tree 2 corrects:   P = 0.75 → Residual = 1 - 0.75 = 0.25
# Tree 3 corrects:   P = 0.82 → Residual = 1 - 0.82 = 0.18
# ...
# Final:             P = 0.91 → above 0.5 → DIABETIC ✅
#
# Same sequential residual correction as regression,
# just using log-odds instead of raw values.
# ============================================================


# ============================================================
# 3. KEY HYPERPARAMETERS FOR CLASSIFICATION
#
# SCALE_POS_WEIGHT:
# This is a LIFESAVER if your data is IMBALANCED
# (e.g. 90% Healthy, 10% Diabetic). Setting this helps
# the model "pay more attention" to the rare cases.
#
# Rule of thumb:
# scale_pos_weight = count(negative class) / count(positive class)
# e.g. 900 healthy, 100 diabetic → scale_pos_weight = 9
#
# MAX_DEPTH:
# How deep each tree goes. For classification, 3 to 7 is
# usually the "Sweet Spot."
#
# GAMMA:
# The "Minimum Loss Reduction" required to make a split.
# Higher gamma → more conservative → less likely to overfit
# ============================================================


# ============================================================
# RANDOM FOREST vs XGBOOST — CLASSIFICATION COMPARISON
#
# ┌──────────────────┬──────────────────┬───────────────────┐
# │ Feature          │ Random Forest    │ XGBoost           │
# ├──────────────────┼──────────────────┼───────────────────┤
# │ Logic            │ Majority vote of │ Sequential        │
# │                  │ deep trees       │ correction of     │
# │                  │                  │ shallow trees     │
# │ Speed            │ Good             │ Extremely fast    │
# │                  │                  │ (parallel)        │
# │ Missing Values   │ Fails — must     │ Built-in, handles │
# │                  │ fill them first  │ NaNs automatically│
# │ Outliers         │ Very robust      │ Sensitive, needs  │
# │                  │                  │ tuning            │
# │ Imbalanced data  │ No built-in fix  │ scale_pos_weight  │
# │                  │                  │ handles it        │
# └──────────────────┴──────────────────┴───────────────────┘
# ============================================================


# ============================================================
# 5. MULTI-CLASS CLASSIFICATION
#
# If you have more than two categories (e.g. "Type 1",
# "Type 2", "No Diabetes"), XGBoost handles this easily.
# Just change the objective:
#
# objective='binary:logistic'  → binary (0 or 1)
# objective='multi:softmax'    → returns class LABEL
#                                 (e.g. 0, 1, or 2)
# objective='multi:softprob'   → returns PROBABILITY for
#                                 each class
#
# Also set num_class=N when using multi:softmax/softprob
# ============================================================


# ============================================================
# CODE
# ============================================================

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 2. Initialize the Classifier
xgb_clf = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='binary:logistic', # tells it to do 0/1 classification
    scale_pos_weight=1,  # increase if data is imbalanced
                         # = count(negative) / count(positive)
    n_jobs=-1
)

# 3. Fit the model
xgb_clf.fit(X_train, y_train)

# 4. Predict
# .predict()       → gives you 0 or 1 (class label)
# .predict_proba() → gives you actual percentages
y_pred = xgb_clf.predict(X_test)
y_prob = xgb_clf.predict_proba(X_test)
# y_prob[:, 1] → probability of class 1 (positive class)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred))


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. XGBoost Classifier uses LOG-ODDS internally, then
#    converts to probabilities via Sigmoid — same residual
#    correction logic as regression, just different math
#
# 2. objective='binary:logistic' → binary classification
#    objective='multi:softmax'   → multiclass (class labels)
#    objective='multi:softprob'  → multiclass (probabilities)
#
# 3. scale_pos_weight → critical for imbalanced datasets
#    Formula: count(negative) / count(positive)
#    Without this, XGBoost ignores the minority class
#
# 4. .predict() → class label (0 or 1)
#    .predict_proba()[:, 1] → probability of positive class
#    Use predict_proba when you need to tune the threshold
#    beyond the default 0.5
#
# 5. XGBoost handles NaN automatically — Random Forest
#    needs manual imputation before training
#
# 6. INTERVIEW TRAP:
#    Q: "How does XGBoost handle imbalanced datasets?"
#    Right answer: "XGBoost has a built-in parameter called
#    scale_pos_weight which multiplies the gradient for
#    the minority class by this factor, making the model
#    pay proportionally more attention to getting those
#    cases right. The standard formula is count(negative
#    class) / count(positive class). For example, if you
#    have 900 healthy and 100 diabetic patients, set
#    scale_pos_weight=9. This is more elegant than
#    oversampling or undersampling because it works within
#    the model's own loss function."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Model Deployment with Streamlit
#    Saving your XGBoost/Random Forest model and deploying
#    it as a web app for your Resume Screener project.
# ============================================================