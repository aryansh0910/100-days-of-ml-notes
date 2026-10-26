# ============================================================
# FEATURE IMPORTANCE — RANDOM FOREST
# ============================================================

# ============================================================
# WHAT IS IT?
#
# Feature importance in a Random Forest is much more
# RELIABLE than in a single Decision Tree. Because it
# averages the importance across hundreds of trees, it is
# less likely to be "tricked" by noise or outliers.
# ============================================================


# ============================================================
# 1. HOW THE FOREST CALCULATES IMPORTANCE
#
# The model keeps a running "SCORE" for every feature:
#
# → Every time a tree uses a feature (like BMI) to split
#   a node, the model looks at how much the "impurity"
#   (variance) DROPPED because of that split.
# → If the split made the data much cleaner, BMI gets a
#   lot of points.
# → The model then AVERAGES those points across all 100+
#   trees in your forest.
# → Finally, it SCALES all scores so they sum up to 1.0.
#
# VISUAL — how scores accumulate across trees:
#
# Feature BMI:
# Tree 1  → used BMI at depth 1, impurity dropped 0.30 → score: 0.30
# Tree 2  → used BMI at depth 2, impurity dropped 0.15 → score: 0.15
# Tree 3  → didn't use BMI at all                      → score: 0.00
# ...
# Tree 100 → average BMI score across all trees = 0.28
#
# All features averaged and scaled so total sums to 1.0:
# BMI: 0.28, S5: 0.18, BP: 0.12, ... (all sum to 1.0)
# ============================================================


# ============================================================
# 2. WHY IT'S BETTER THAN A SINGLE TREE
#
# THE "GREEDY" PROBLEM:
# A single tree might get "obsessed" with one feature and
# ignore others. A Random Forest FORCES different trees to
# look at different features (via max_features), giving you
# a more complete picture of what matters.
#
# STABILITY:
# If you change a few rows of data, a single tree's
# importance might flip-flop completely. A Random Forest's
# importance remains very STABLE.
#
# VISUAL — single tree vs forest importance:
#
# Single Tree (unstable, greedy):
# BMI:    0.50  ← dominates because it's always picked first
# Age:    0.30
# BP:     0.15
# Others: 0.05
#
# Random Forest (balanced, reliable):
# BMI:    0.28  ← still important but not inflated
# S5:     0.18
# BP:     0.15
# Age:    0.12
# Others: 0.27  ← hidden patterns now visible!
# ============================================================


# ============================================================
# 4. WHAT THE RESULTS USUALLY SHOW
#
# In the Diabetes dataset, you will almost always see BMI
# and S5 (a blood serum measurement) at the very top.
#
# If a feature has a score of 0.30 → that feature is
# responsible for 30% of the total "uncertainty reduction"
# in the forest.
#
# If a feature has a score near 0.00 → the forest
# basically IGNORED it because it didn't help make better
# predictions.
# ============================================================


# ============================================================
# 5. PROFESSIONAL TIP — "DROP COLUMN" IMPORTANCE
#
# If you really want to be a pro, compare the Random Forest
# importance with the Lasso results you did earlier.
#
# If Lasso says a feature is 0 (no linear relationship)...
# But Random Forest says it is 0.15 (high importance)...
#
# CONCLUSION: That feature has a powerful NON-LINEAR or
# INTERACTION effect that a simple linear model is too
# "blind" to see.
#
# VISUAL — Lasso vs Random Forest comparison:
#
# Feature     Lasso Coef   RF Importance   Conclusion
# ─────────────────────────────────────────────────────────
# BMI         0.45         0.28            ✅ linear + non-linear
# S5          0.00         0.18            ⚠️ non-linear only!
# Age         0.12         0.11            ✅ mostly linear
# City_xyz    0.00         0.00            ❌ genuinely useless
#
# S5 is the interesting case — Lasso zeroed it out (no
# straight-line relationship) but Random Forest found it
# important (complex interaction with other features).
# ============================================================


# ============================================================
# CODE
# ============================================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
import pandas as pd
import matplotlib.pyplot as plt

diabetes = load_diabetes()
X_reg = diabetes.data
y_reg = diabetes.target

# 1. Train the Forest
rf = RandomForestRegressor(n_estimators=100,
                            max_depth=6,
                            random_state=42)
rf.fit(X_reg, y_reg)

# 2. Extract Importance
# .feature_importances_ is the attribute that holds scores
importance_scores = rf.feature_importances_

# 3. Create a clean DataFrame
feature_importance_df = pd.DataFrame({
    'Feature'   : diabetes.feature_names,
    'Importance': importance_scores
}).sort_values(by='Importance', ascending=False)

# 4. Print the result
print(feature_importance_df)
# Scores sum to 1.0 — each score = % of total impurity
# reduction that feature was responsible for

# 5. Visualize
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'],
         feature_importance_df['Importance'],
         color='teal')
plt.xlabel('Importance Score')
plt.title('Feature Importance: Random Forest (Diabetes Dataset)')
plt.gca().invert_yaxis()  # Highest importance at the top
plt.show()


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Feature importance = average impurity reduction across
#    all trees, scaled to sum to 1.0
#
# 2. More reliable than single tree because:
#    → Random forest forced trees to look at different
#      features (max_features) → no one feature dominates
#    → Averaged across 100+ trees → stable, not affected
#      by a few noisy rows
#
# 3. Score of 0.30 → that feature explains 30% of all
#    uncertainty reduction in the forest
#    Score near 0.00 → forest ignored it, genuinely useless
#
# 4. PROFESSIONAL TIP: compare with Lasso results
#    Lasso=0, RF=high → non-linear / interaction effect
#    Lasso=high, RF=high → strong linear + non-linear
#    Both=0 → drop it, completely useless
#
# 5. INTERVIEW TRAP:
#    Q: "Why is Random Forest feature importance more
#        reliable than a single Decision Tree's?"
#    Right answer: "A single tree is greedy — it always
#    picks the strongest feature first at the root, giving
#    it an inflated importance score. Other features rarely
#    get a chance. Random Forest forces DIFFERENT features
#    to be considered at each split via max_features, so
#    every feature gets a fair chance across hundreds of
#    trees. The averaged importance is stable and unbiased,
#    unlike a single tree which can completely flip its
#    importance rankings if you change a few rows."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Boosting — AdaBoost
#    Sequential ensemble where each model focuses on the
#    mistakes of the previous one.
# ============================================================