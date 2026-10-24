# ============================================================
# DECISION TREE VISUALIZATION — dtreeviz
# ============================================================

# ============================================================
# WHAT IS IT?
#
# USE THIS TO CHECK OVERFITTING — THAT IT DOESN'T DRAW
# UNNECESSARY SPLITS IN THE TREE.
#
# dtreeviz is a library used to visualize our decision
# trees for better understanding the data. We can also see
# the trees using the Decision Tree's inbuilt feature in
# Scikit-Learn, but it's not so attractive so we use the
# dtreeviz library.
#
# Unlike the standard plot_tree from Scikit-Learn (which
# is just boxes and text), dtreeviz shows the DISTRIBUTIONS
# of your data at every single split.
# ============================================================


# ============================================================
# WHY USE dtreeviz OVER THE DEFAULT?
#
# HISTOGRAMS AT NODES:
# It shows a histogram of your features at every split,
# so you can see where the "Yes/No" cut actually falls in
# the data distribution.
#
# FEATURE SPACE VISUALIZATION:
# It shows how the classes are overlapping or separating.
#
# PATH HIGHLIGHTING:
# You can pass a single "row" (like one specific customer)
# through the tree, and it will highlight the EXACT PATH
# taken to reach a prediction.
#
# VISUAL — default sklearn vs dtreeviz:
#
# sklearn plot_tree (boring):
# ┌────────────────────┐
# │ petal_length <= 2.5│
# │ gini = 0.667       │
# │ samples = 150      │
# └────────────────────┘
#       /         \
#    [box]       [box]   ← just numbers, hard to interpret
#
# dtreeviz (informative):
# ┌────────────────────────────────┐
# │ petal_length <= 2.5            │
# │ [histogram showing where 2.5   │
# │  falls in the actual data,     │
# │  colored by class]             │
# └────────────────────────────────┘
#       /                \
# [histogram]        [histogram]  ← you can SEE the split!
# ============================================================


# ============================================================
# CODE — CLASSIFICATION TREE (IRIS)
# ============================================================

import dtreeviz
from sklearn.datasets import load_iris, load_diabetes
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# 1. Load data
iris = load_iris()
diabetes = load_diabetes()
X = iris.data
y = iris.target

# 2. Train a simple tree
clf = DecisionTreeClassifier(max_depth=5,
                              random_state=42,
                              min_samples_leaf=5)
clf.fit(X, y)

# 3. Create the dtreeviz object
viz_model = dtreeviz.model(
    clf,
    X_train=X,
    y_train=y,
    target_name="target",         # name of the output column
    feature_names=iris.feature_names  # names of all headers
)

# 4. Display or Save
# In a Jupyter Notebook:
# viz_model.view()

# To save as an SVG to open in VS Code:
viz_model.view().save("decision_tree_viz.svg")


# ============================================================
# CODE — REGRESSION TREE (DIABETES)
#
# SIMILARLY U CAN USE THIS FOR NUMERICAL DATA AS WELL.
# THEN IT WILL SHOW U THE CUTS ON THE SCATTER PLOT.
# ============================================================

reg = DecisionTreeRegressor(max_depth=3, min_samples_leaf=5)

X = diabetes.data    # this will get you the input columns
y = diabetes.target  # this will get you the output column

reg.fit(X, y)

viz_model1 = dtreeviz.model(
    reg,
    X_train=X,
    y_train=y,
    target_name="progression of the diabetes",
    feature_names=diabetes.feature_names
)

viz_model1.view().save("decision_tree_regressor_viz.svg")

# For regression, instead of histograms colored by class,
# dtreeviz shows SCATTER PLOTS at each node — you can
# literally SEE the staircase cuts being made on your data


# ============================================================
# HOW TO USE PATH HIGHLIGHTING — BONUS
# ============================================================

# Pass a single row to see exactly how the tree classifies it
# viz_model.view(X=X[0])  ← highlights the path for row 0
#
# This is extremely useful for:
# → Explaining a specific prediction to a non-technical boss
# → Debugging why the model made a wrong prediction
# → Checking if the tree logic makes business sense


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. dtreeviz → use this over sklearn's plot_tree for
#    proper visualization with histograms at each node
#
# 2. Three main advantages:
#    → Histograms at nodes (see WHERE the split falls)
#    → Class overlap visualization
#    → Path highlighting for individual predictions
#
# 3. For classification → shows histograms colored by class
#    For regression → shows scatter plots with cut lines
#
# 4. Use it specifically to CHECK OVERFITTING — if the tree
#    has too many tiny splits that barely separate the data,
#    you can SEE it visually and adjust max_depth or
#    min_samples_leaf accordingly
#
# 5. Save as .svg for VS Code or .html for browser viewing
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Random Forest
#    Combining many Decision Trees into an ensemble to
#    reduce overfitting and improve accuracy significantly.
# ============================================================