# ============================================================
# PCA — PRINCIPAL COMPONENT ANALYSIS
# ============================================================

# ============================================================
# WHAT IS IT?
#
# PCA ==> ITS A UNSUPERVISED TECHNIQUE, ITS A FEATURE
# EXTRACTION TECHNIQUE.
#
# Geometrically, you have to stop thinking about "deleting
# columns" and start thinking about "changing your
# perspective."
#
# Imagine you are looking at a 3D object, but you want to
# capture its essence in a 2D photograph. PCA is the
# process of finding the BEST ANGLE to take that photo.
# ============================================================


# ============================================================
# THE CORE IDEA — MAXIMIZING VARIANCE
#
# If you have a cloud of data points, PCA looks for a new
# axis (a line) where the data is the most "spread out."
#
# WHY? Because spread (variance) equals information.
# If all your points are squished together on an axis, you
# can't tell them apart. If they are spread out, the
# differences between customers (like Churn vs No Churn)
# become clear.
#
# ITS BENEFIT IS THAT ALGO RUNS FASTER AS THE FEATURES
# WOULD BE LESS.
#
# FEATURES ARE DECIDED BASED ON THEIR VARIANCE — OR U CAN
# SAY THE SPREAD. MORE IS THE SPREAD, MORE IS THE INFO IT
# CONTAINS IN IT.
#
# U HAVE TO STANDARDIZE UR DATA AT FIRST TO USE PCA.
#
# WHY STANDARDIZE FIRST?
# PCA picks the axis with maximum variance. If Salary
# (range 20,000-150,000) and Age (range 18-80) are not
# scaled, Salary's variance will be HUGE just because of
# its bigger numbers — not because it's actually more
# important. PCA would wrongly think Salary is the most
# "informative" feature. Scaling first puts both on equal
# footing.
#
# VISUAL — finding the best axis:
#
# Original data cloud (2D):
#
#        y
#        |    ●
#        |  ●   ●
#        |    ●     ●
#        |  ●    ●
#        |________________ x
#
# PCA finds the line where points are MOST spread out:
#
#        y
#        |    ●        ╱ ← PC1 (max variance direction)
#        |  ●   ●    ╱
#        |    ●    ╱●
#        |  ●    ╱  ●
#        |_____╱__________ x
#
# Projecting all points onto this single line (PC1)
# captures most of the original information — even though
# we went from 2D to 1D!
# ============================================================


# ============================================================
# CODE — USING PCA IN A PIPELINE
# ============================================================

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Let's say we want to squash our many columns into just
# 3 'Principal Components'
pca_step = PCA(n_components=3)

# Update your pipe (MUST be after StandardScaler)
pipe = Pipeline([
    ("transforming 1", trf1),
    ("Transforming 2", trf2),
    ("Transformer", trf3),  # Scaling is 100% required for PCA
    ("pca", pca_step),      # PCA goes here
    ("MODEL", model)
])

# pca.explained_variance_
# this will give you variance of each component


# ============================================================
# HOW TO FIND OPTIMUM NUMBER OF COMPONENTS
#
# EACH COMPONENT'S VARIANCE WILL GIVE YOU THE AMOUNT OF
# DATA RETAINED BY EACH COMPONENT.
#
# IN GENERAL, U HAVE TO GET AT LEAST 90 PERCENT OF THE
# TOTAL VARIANCE RETAINED.
#
# HOW TO DO THIS IN PRACTICE:
# Use explained_variance_ratio_ — gives % of total variance
# each component captures, then add them up cumulatively
# until you hit 90% (or 95%, depending on the project).
#
# VISUAL — Scree plot logic:
#
# Component   Variance %   Cumulative %
# PC1         45%          45%
# PC2         25%          70%
# PC3         18%          88%
# PC4          7%          95%  ← stop here for 95% threshold
# PC5          3%          98%
# PC6          2%         100%
#
# So instead of using all 6 components, you'd pick n=4
# to retain 95% of the information with 1/3 fewer features!
# ============================================================

import numpy as np

# Example workflow to find optimum components
pca_full = PCA()
# pca_full.fit(X_scaled)

# Cumulative variance explained
# cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
# print(cumulative_variance)

# Find how many components needed for 90% variance
# n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
# print(f"Components needed for 90% variance: {n_components_90}")


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. PCA = unsupervised, feature extraction technique
#    Doesn't use target variable y at all
#
# 2. Core idea: find the axis where data has MAXIMUM
#    variance — that axis (Principal Component) captures
#    the most information
#
# 3. MUST standardize data before PCA — otherwise high-range
#    columns dominate just because of their scale, not
#    their actual importance
#
# 4. Benefit: faster algorithms (fewer features), less
#    overfitting, fights curse of dimensionality
#
# 5. Finding optimum components:
#    Use explained_variance_ratio_, take cumulative sum,
#    pick n where cumulative % crosses 90% (or chosen threshold)
#
# 6. PCA must come AFTER scaling but BEFORE the model in
#    a pipeline
#
# 7. INTERVIEW TRAP:
#    Q: "Does PCA select the most important original
#        features?"
#    Wrong: "Yes, it picks the best columns"
#    Right: "No — PCA doesn't select existing columns, it
#    CREATES new ones (linear combinations of all original
#    columns) along directions of maximum variance. The
#    original column names/meaning are lost; you get PC1,
#    PC2, etc. instead. That's feature EXTRACTION, not
#    SELECTION."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Feature Construction & Feature Splitting
#    Creating new meaningful columns by combining or
#    splitting existing ones for better model signal.
# ============================================================