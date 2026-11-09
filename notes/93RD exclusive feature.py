# ============================================================
# EFB — EXCLUSIVE FEATURE BUNDLING
# ============================================================

# ============================================================
# WHAT IS IT?
#
# Think of Exclusive Feature Bundling (EFB) as a clever way
# to "PACK A SUITCASE" more efficiently so the model has
# less to carry.
#
# In many datasets — especially after you use OneHotEncoder
# or get_dummies — you end up with a lot of columns that
# are MOSTLY ZEROS. These are called SPARSE FEATURES.
# ============================================================


# ============================================================
# THE PROBLEM
#
# THE MAIN PROBLEM IS IF ANY COLUMN IS 1 THEN THE OTHERS
# SHOULD BE 0. EXAMPLE:
#
# A car can't be Petrol AND Diesel at the same time. If
# it's Petrol (1), the others must be zero. These features
# are "EXCLUSIVE" — they never have values at the same time.
#
# Processing THREE SEPARATE COLUMNS is a waste of time
# for the computer.
#
# VISUAL — sparse columns after OHE:
#
# ┌───────┬────────┬────────┬───────┐
# │ Row   │ Petrol │ Diesel │ LPG   │
# ├───────┼────────┼────────┼───────┤
# │ Car A │   1    │   0    │   0   │ ← 2 wasted zeros
# │ Car B │   0    │   1    │   0   │ ← 2 wasted zeros
# │ Car C │   0    │   0    │   1   │ ← 2 wasted zeros
# │ Car D │   1    │   0    │   0   │ ← 2 wasted zeros
# └───────┴────────┴────────┴───────┘
# Computer processes 12 values but only 4 are useful!
# ============================================================


# ============================================================
# THE SOLUTION — BUNDLING
#
# LightGBM realizes that since these columns NEVER "OVERLAP,"
# it can safely BUNDLE them into a SINGLE COLUMN without
# losing any information.
#
# HOW IT WORKS:
#
# STEP 1 — IDENTIFY:
# Look for features that are RARELY NON-ZERO at the
# same time (exclusive features).
#
# STEP 2 — OFFSET:
# Add a "buffer" or OFFSET to the values so they don't
# crash into each other.
#
# Feature A has values [0, 1]        → stays as [0, 1]
# Feature B has values [0, 1] + 2    → becomes [2, 3]
# Feature C has values [0, 1] + 4    → becomes [4, 5]
#
# STEP 3 — MERGE:
# Put them all in ONE COLUMN.
# If value = 1 → was Feature A (Petrol)
# If value = 3 → was Feature B (Diesel)
# If value = 5 → was Feature C (LPG)
#
# LightGBM decides to keep Petrol values as they are
# (0 and 1), but it SHIFTS Diesel values by adding a
# "buffer" (an offset). It says: "I'll add 2 to every
# Diesel value."
# Original Diesel 0 becomes 2.
# Original Diesel 1 becomes 3.
# ============================================================


# ============================================================
# BEFORE AND AFTER — EFB
#
# BEFORE:
# ┌───────┬──────────────────┬──────────────────┐
# │ Row   │ Column1 (Petrol) │ Column2 (Diesel) │
# ├───────┼──────────────────┼──────────────────┤
# │ Car A │        1         │        0         │
# │ Car B │        0         │        1         │
# │ Car C │        1         │        0         │
# └───────┴──────────────────┴──────────────────┘
# 2 columns, 6 values processed, 3 are wasted zeros
#
# AFTER:
# ┌───────┬──────────────────────┐
# │ Row   │ Column_fuel (Bundle) │
# ├───────┼──────────────────────┤
# │ Car A │          1           │  ← was Petrol=1
# │ Car B │          3           │  ← was Diesel=1 (+2 offset)
# │ Car C │          1           │  ← was Petrol=1
# └───────┴──────────────────────┘
# 1 column, 3 values processed, ZERO wasted! ✅
#
# IT INCREASES EFFICIENCY BY DECREASING THE NO. OF COLUMNS
# ============================================================


# ============================================================
# WHY THIS IS POWERFUL IN REAL DATASETS
#
# After OHE on a "City" column with 500 cities:
# → 500 columns, each row has 499 zeros and 1 one
# → Computer processes 500 values per row, 499 useless
#
# After EFB:
# → 1 column with offset values
# → Computer processes 1 value per row ✅
# → 500x less memory, 500x fewer split candidates!
#
# This is why LightGBM doesn't need OHE — EFB handles
# the sparse columns that OHE creates far more efficiently.
# ============================================================


# ============================================================
# GOSS + EFB + BINNING — HOW ALL THREE WORK TOGETHER
#
# ┌──────────────┬────────────────────────────────────────┐
# │ Trick        │ What it reduces                        │
# ├──────────────┼────────────────────────────────────────┤
# │ Binning      │ Number of SPLIT POINTS per feature     │
# │              │ 10,000 values → 255 bins               │
# │ GOSS         │ Number of ROWS processed per tree      │
# │              │ 500 rows → 140 rows (top errors only)  │
# │ EFB          │ Number of COLUMNS processed            │
# │              │ 500 OHE columns → 1 bundled column     │
# └──────────────┴────────────────────────────────────────┘
#
# Together they reduce computation in ALL THREE dimensions:
# rows, columns, and split points → 2-10x faster than GBM
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. EFB targets SPARSE FEATURES — columns that are mostly
#    zeros and never non-zero at the same time (exclusive)
#
# 2. Bundling adds an OFFSET to each feature's values so
#    they occupy different ranges in one column:
#    Petrol: [0,1] → stays [0,1]
#    Diesel: [0,1] → becomes [2,3]
#    LPG:   [0,1] → becomes [4,5]
#
# 3. LightGBM can decode the bundle back perfectly —
#    no information is lost, just packed more efficiently
#
# 4. This is the main reason LightGBM doesn't need OHE —
#    EFB internally handles what OHE creates externally
#    but in a MUCH more memory-efficient way
#
# 5. INTERVIEW TRAP:
#    Q: "How does LightGBM handle high-cardinality
#        categorical features without OHE?"
#    Right answer: "Two mechanisms. First, it accepts
#    categorical columns directly via categorical_feature
#    and finds optimal group splits internally. Second,
#    if OHE was already applied, EFB (Exclusive Feature
#    Bundling) automatically detects that the resulting
#    sparse columns are mutually exclusive and bundles
#    them into a single column using offsets — so 500
#    OHE city columns become 1 bundled column. Both
#    approaches eliminate the memory explosion that OHE
#    causes in other algorithms."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → CatBoost
#    Another gradient boosting library specifically built
#    for categorical features — uses ordered boosting to
#    avoid target leakage during encoding.
# ============================================================