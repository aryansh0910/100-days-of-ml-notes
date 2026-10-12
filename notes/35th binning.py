# ============================================================
# BINNING AND BINARIZATION — GROUPING NUMERICAL DATA
# ============================================================

# ============================================================
# WHAT IS IT?
# Sometimes raw numerical values (like Age: 21, 22, 23...)
# give the model TOO MUCH detail and it overfits.
# Binning groups these values into MEANINGFUL BUCKETS.
#
# REAL WORLD EXAMPLE:
# LIKE IF THERE ARE VALUES IN TENS OF THOUSANDS AND LAKHS,
# YOUR COMPUTER MIGHT OVERFIT SO YOU HAVE TO CREATE BINS
# LIKE 1-1000 THEN FROM 1000-2000 LIKE THIS.
# ============================================================


# ============================================================
# PART 1 — BINNING (KBinsDiscretizer)
# ============================================================
#
# Binning = turning a CONTINUOUS variable (like Age)
# into DISCRETE "bins" or intervals
# (like Child, Teen, Adult, Senior)
#
# WHY USE IT?
# A Decision Tree can OVERFIT if it tries to find a pattern
# for EVERY single age (21, 22, 23...).
# By grouping them → simplifies the pattern → more robust!
#
# TWO CATEGORIES OF BINNING:
#
# 1. UNSUPERVISED BINNING (KBinsDiscretizer)
#    → Doesn't use the target variable y to decide bins
#    → Three strategies: Uniform, Quantile, KMeans
#
# 2. SUPERVISED BINNING (Decision Tree Binning)
#    → Uses the target variable y to find BEST bin boundaries
#    → Smarter — bins are chosen to maximize prediction power
#
# 3. CUSTOM BINNING (Based on domain knowledge)
#    → YOU manually define the boundaries
#    → Example: pd.cut(df['age'], bins=[0,18,35,60,100])

# ── THREE STRATEGIES FOR KBinsDiscretizer ────────────────────
#
# UNIFORM:
# → All bins have the SAME WIDTH
# → Example for Age 0-100 with 5 bins:
#   [0-20] [20-40] [40-60] [60-80] [80-100]
# → Problem: some bins may have very few people
#
# QUANTILE (Most Common ✅):
# → All bins have the SAME NUMBER OF DATA POINTS
# → Example: 20% of people in each bin
# → Each bin is equally "populated"
# → Best choice for most cases!
#
# KMEANS:
# → Uses KMeans clustering to find bin boundaries
# → Bins are defined by natural clusters in the data

# ── ENCODE PARAMETER ─────────────────────────────────────────
#
# TO USE UNSUPERVISED BINNING WE USE KBinsDiscretizer()
# WITH THREE PARAMETERS:
# 1. n_bins   → how many bins to create
# 2. strategy → uniform, quantile, or kmeans
# 3. encode   → how to represent the bins as numbers
#
# encode='ordinal':
# → Gives numbers (0, 1, 2, 3, 4...) to each bin
# → Bin 1 (youngest) = 0, Bin 5 (oldest) = 4
# → USE THIS when bins have natural ORDER (like age groups)
#
# encode='onehot':
# → Creates separate column for EACH bin (like get_dummies)
# → USE THIS when bins have NO natural order


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, Binarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ============================================================
# UNSUPERVISED BINNING — KBinsDiscretizer IN ACTION
# ============================================================

# strategy='quantile' → equal number of passengers per bin
# encode='ordinal'    → gives numbers (0,1,2,3,4) to each bin
#                       instead of many One-Hot columns
trf_bin = ColumnTransformer([
    ('age_bin',
     KBinsDiscretizer(n_bins=5,
                      encode='ordinal',
                      strategy='quantile'),
     ['age'])
],
remainder='passthrough')

# WHAT THIS DOES TO AGE COLUMN:
# Raw Age: [1, 5, 16, 22, 35, 42, 55, 70, 80]
# After binning into 5 groups:
# [0, 0, 1, 1, 2, 2, 3, 4, 4]
# → Bin 0 = youngest group (children)
# → Bin 4 = oldest group (seniors)
# Now the model sees 5 GROUPS instead of hundreds of ages!


# ============================================================
# QUICK DEMO — SEE THE BINNING IN ACTION
# ============================================================

sample_ages = np.array([[1], [16], [22], [35],
                         [42], [55], [70], [80]])

kbd = KBinsDiscretizer(n_bins=4,
                       encode='ordinal',
                       strategy='quantile')
binned_ages = kbd.fit_transform(sample_ages)

print("Original Ages → Binned Values:")
for original, binned in zip(sample_ages, binned_ages):
    print(f"  Age {original[0]:3d} → Bin {int(binned[0])}")

print(f"\nBin edges: {kbd.bin_edges_}")
# bin_edges_ shows you exactly WHERE each bin starts/ends


# ============================================================
# PART 2 — BINARIZATION
# ============================================================
#
# Binarization = extreme version of binning.
# Takes any numerical value → turns it into EXACTLY 0 or 1
# based on a THRESHOLD you set.
#
# WHY USE IT?
# Imagine SibSp (Siblings/Spouses on Titanic).
# You don't care if someone has 1, 2, or 5 siblings.
# You ONLY care: "Are they ALONE (0) or WITH FAMILY (1)?"
#
# SibSp = 0 → 0 (travelling alone)
# SibSp = 1 → 1 (travelling with family)
# SibSp = 3 → 1 (travelling with family)
# SibSp = 5 → 1 (travelling with family)
#
# THRESHOLD = 0 means:
# value > 0  → 1 (has at least one sibling/spouse)
# value <= 0 → 0 (travelling alone)

binarizer = Binarizer(threshold=0)
# threshold=0 → any value ABOVE 0 becomes 1
#               any value AT or BELOW 0 becomes 0

sample_sibsp = np.array([[0], [1], [2], [5], [0], [3]])
binarized = binarizer.fit_transform(sample_sibsp)

print("\n\nBinarization Demo (SibSp):")
print("SibSp → IsWithFamily")
for original, binary in zip(sample_sibsp, binarized):
    status = "With Family" if binary[0] == 1 else "Alone"
    print(f"  {original[0]} → {int(binary[0])} ({status})")


# ============================================================
# BINNING vs BINARIZATION — WHEN TO USE WHICH?
# ============================================================
#
# ┌──────────────┬────────────────────────────────────────┐
# │              │ Use When                               │
# ├──────────────┼────────────────────────────────────────┤
# │ Binning      │ You want MULTIPLE groups (age groups,  │
# │              │ income brackets, price ranges)         │
# ├──────────────┼────────────────────────────────────────┤
# │ Binarization │ You only care about YES/NO, HAS/HASN'T │
# │              │ (alone vs family, active vs inactive)  │
# └──────────────┴────────────────────────────────────────┘


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Binning = grouping continuous values into BUCKETS
#    Prevents overfitting on granular numerical values
#
# 2. THREE strategies for KBinsDiscretizer:
#    Uniform  → equal width bins
#    Quantile → equal number of samples per bin ✅ (use this!)
#    KMeans   → cluster-based bin boundaries
#
# 3. encode='ordinal'  → gives numbers 0,1,2... to bins
#    encode='onehot'   → creates separate column per bin
#    Use ordinal when bins have natural ORDER (age groups)
#
# 4. Binarization = extreme binning → ONLY 0 or 1
#    Use when you just need YES/NO (alone vs with family)
#
# 5. TWO categories of binning:
#    Unsupervised → KBinsDiscretizer (doesn't use y)
#    Supervised   → Decision Tree Binning (uses y)
#    Custom       → pd.cut() with manual boundaries
#
# 6. Use binning INSIDE a Pipeline via ColumnTransformer
#    Same golden rule: fit on train, transform on test!
# ============================================================


# ============================================================
# COMING UP NEXT:
# → ML Algorithms
#    Preprocessing is now FULLY covered!
#    Time to actually train models — Linear Regression,
#    Logistic Regression, Decision Trees and more!
# ============================================================