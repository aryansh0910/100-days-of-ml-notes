# ============================================================
# LIGHTGBM — DEEP DIVE: LEAF-WISE, GOSS, AND BINNING
# ============================================================

# ============================================================
# LEAF-WISE GROWTH — THE CORE IDEA
#
# Most models (like XGBoost) grow their decision trees like
# a professional gardener trims a hedge — LEVEL BY LEVEL,
# making sure everything is perfectly even and balanced.
#
# LightGBM is different. It grows "LEAF-WISE." It looks at
# the whole tree and asks: "Which specific BRANCH is causing
# the most errors?" Then it only grows THAT ONE BRANCH,
# even if the tree looks totally lopsided or "ugly."
#
# WHY DOES THIS MATTER? It cuts down the error much faster
# than a balanced tree because it doesn't waste time on the
# "easy" parts of the data.
#
# IT PROVIDES BETTER ACCURACY AS IT SPENDS ITS MAJORITY OF
# THE TIME IN CORRECTING ITS MISTAKES RATHER THAN SEEING
# WHAT IS ALREADY ACCURATE.
#
# VISUAL — Level-wise vs Leaf-wise:
#
# XGBoost (Level-wise, wastes time on easy branches):
#            [Root]
#           /      \
#        [N1]      [N2]     ← grows BOTH even if N2 is fine
#       /    \    /    \
#     [L1] [L2] [L3] [L4]  ← grows ALL 4 even if L3,L4 fine
#
# LightGBM (Leaf-wise, focuses on worst branch only):
#            [Root]
#           /      \
#        [N1]      [L3]  ← N2 has low error, STOP here
#       /    \
#     [N2]  [L2]         ← N1 had high error, keep going
#    /    \
#  [L1]  [L4]            ← grows DEEP only where needed ✅
#
# Result: Same number of leaves, but LightGBM's tree
# captures complex patterns where errors are highest.
# ============================================================


# ============================================================
# GOSS — GRADIENT BASED ONE SIDE SAMPLING
#
# THIS IS KNOWN AS GOSS ==> GRADIENT BASED ONE SIDE
# SAMPLING ==> MEANS FOCUSING ON THE WRONG DATA.
#
# LightGBM uses a trick called GOSS. If your model already
# perfectly predicts the price of 80% of your cars,
# LightGBM says: "I'm bored with these easy cars. I'm
# going to spend 90% of my energy focusing on the 20% of
# cars that I keep getting wrong."
#
# By focusing on its MISTAKES rather than the whole dataset,
# it finishes the "learning" process in RECORD TIME.
#
# DETAILED EXPLANATION:
# Assume a sample of 500 rows and base model M0.
# It will have 500 GRADIENTS (residuals) = errors.
#
# LightGBM will:
# STEP 1 → Arrange those gradients in DESCENDING ORDER
# STEP 2 → Select TOP 20% with HIGHEST gradient (error)
#           to train the next model
# STEP 3 → Randomly select 10% from the REMAINING 80%
#           (low gradient rows)
# STEP 4 → Train M1 (next model) on this combined sample
#
# SO THE SAMPLING IS DONE ONLY FROM ONE SIDE I.E THE
# 10% VERSION I.E LOW GRADIENTS — hence "ONE SIDE SAMPLING"
#
# VISUAL — GOSS sampling:
#
# 500 rows sorted by gradient (error) high → low:
#
# ████████████████████  ← Top 20% (100 rows) HIGH ERROR
# ████████████████████     ALL selected ✅
# ████████████████████
# ────────────────────
# ░░░░░░░░░░░░░░░░░░░░  ← Bottom 80% (400 rows) LOW ERROR
# ░░░░░░░░░░░░░░░░░░░░     Only 10% randomly selected
# ░░░░░░░░░░░░░░░░░░░░     = 40 rows ✅
# ░░░░░░░░░░░░░░░░░░░░
#
# Next model trains on: 100 + 40 = 140 rows instead of 500
# → 3.5x faster, same focus on hard cases ✅
# ============================================================


# ============================================================
# 4. WHEN SHOULD YOU USE IT?
#
# USE IT WHEN:
# You have a MASSIVE DATASET (10,000+ rows) and you're
# tired of waiting for XGBoost to finish.
#
# DON'T USE IT WHEN:
# You have a VERY SMALL DATASET (like only 100 rows).
# Because it grows so aggressively on specific branches,
# it can easily OVERFIT (memorize the data instead of
# learning it) if the dataset is too small.
# ============================================================


# ============================================================
# WHY IS IT FAST — BIN-WAY OF SPLITTING
#
# IT DOESN'T CHECK EVERYTHING TO SPLIT THE DATA. IT JUST
# MAKES BINS FOR THE DATA.
# THIS IS KNOWN AS BIN WAY OF SPLITTING.
#
# Instead of looking at every exact number, LightGBM
# groups them into "BINS" (like buckets).
#
# It says: "I don't care about the difference between
# 12,451 and 12,452. I'm just going to create 255 buckets.
# Bucket 1: 0 to 1,000 km
# Bucket 2: 1,001 to 2,000 km
# ...and so on."
#
# Now instead of checking 10,000 individual numbers, the
# computer only has to check 255 BUCKETS.
#
# VISUAL — exact values vs bins:
#
# WITHOUT binning (XGBoost checks every value):
# Split candidates: 12451, 12452, 12453, 12454 ... 22000
# → 10,000 split points to check ❌ slow
#
# WITH binning (LightGBM checks buckets):
# Bin 1: [0-1000]    Bin 2: [1001-2000]  ...  Bin 255: [...]
# → Only 255 split points to check ✅ fast!
#
# WHY IS THIS BETTER?
#
# MASSIVE SPEED: Checking 255 buckets is much faster than
# checking 10,000 exact numbers.
#
# LESS MEMORY (RAM): The computer doesn't need to remember
# every exact decimal point — just which bucket the car
# belongs to.
#
# REDUCES NOISE: Tiny differences in mileage (like 1 km)
# usually don't change the price of a car. By "binning"
# them, the model focuses on the BIGGER PICTURE instead of
# getting distracted by tiny details.
# ============================================================


# ============================================================
# ALL THREE TRICKS TOGETHER
#
# ┌──────────────────┬──────────────────────────────────────┐
# │ Trick            │ What it does                         │
# ├──────────────────┼──────────────────────────────────────┤
# │ Leaf-wise growth │ Only grows the branch with the       │
# │                  │ highest error → faster convergence   │
# │ GOSS             │ Only trains on high-error rows +     │
# │                  │ small random sample of easy rows     │
# │                  │ → fewer rows to process per tree     │
# │ Binning          │ Groups continuous values into 255    │
# │                  │ buckets → fewer split points to      │
# │                  │ check per feature                    │
# └──────────────────┴──────────────────────────────────────┘
#
# Combined effect: LightGBM is 2-10x faster than XGBoost
# while often being MORE accurate on large datasets.
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Leaf-wise growth → grows the single worst branch
#    instead of all branches at each level → faster error
#    reduction, but overfits on small data
#
# 2. GOSS = top 20% high-error rows (all kept) + 10%
#    random sample of low-error rows → trains on ~30%
#    of data per tree instead of 100%
#
# 3. Binning = 255 buckets instead of exact values →
#    fewer split candidates → faster AND less noise
#
# 4. These three tricks TOGETHER make LightGBM 2-10x
#    faster than XGBoost
#
# 5. Use LightGBM for 10k+ rows. Avoid on small datasets
#    (leaf-wise growth overfits aggressively)
#
# 6. INTERVIEW TRAP:
#    Q: "How does LightGBM achieve its speed over XGBoost?"
#    Right answer: "Three specific tricks. First, LEAF-WISE
#    growth — instead of growing all branches at the same
#    depth, it only grows the single branch with the
#    highest loss reduction, converging faster. Second,
#    GOSS — it trains only on the top 20% high-gradient
#    rows plus a 10% random sample of easy rows, so each
#    tree sees ~30% of data instead of 100%. Third,
#    HISTOGRAM BINNING — instead of checking every unique
#    value as a split point, it groups values into 255
#    bins, reducing split candidates from thousands to 255.
#    Combined, these make LightGBM 2-10x faster while
#    maintaining or improving accuracy on large datasets."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → CatBoost
#    Another gradient boosting library specifically built
#    for datasets with many categorical features — no
#    encoding needed at all.
# ============================================================