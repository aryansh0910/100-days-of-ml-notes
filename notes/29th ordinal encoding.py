# ============================================================
# ORDINAL ENCODING — DEEP DIVE
# ============================================================

# ============================================================
# WHAT IS IT?
#
# USE LABEL ENCODER IN THE Y COLUMN I.E THE OUTPUT OR THE
# COLUMN TO BE PREDICTED AND USE ORDINAL ENCODER IN THE
# X COLUMN.
#
# Why not just use Label Encoding for X?
# (This is a favorite interview question)
#
# LabelEncoder:
# → Designed for the Target variable (y)
# → Assigns numbers based on ALPHABETICAL order
# → You cannot control which category is "higher"
# → Example: Excellent=0, Good=1, Poor=2 (alphabetical)
#             This tells the model Poor > Good > Excellent ❌
#
# OrdinalEncoder:
# → Designed for Input features (X)
# → You MANUALLY pass the order you want
# → Machine knows Masters is higher than Bachelors
# → IN ORDINAL ENCODING WE WILL JUST GIVE THE VALUE TO
#   THE COMPUTER AND TELL IT TO RANK IT HIGHER
#
# VISUAL:
#
# LabelEncoder (alphabetical — WRONG for ordinal data):
# Excellent → 0
# Good      → 1
# Poor      → 2   ← model thinks Poor is the "biggest"! ❌
#
# OrdinalEncoder (your defined order — CORRECT):
# Poor      → 0
# Average   → 1
# Good      → 2
# Excellent → 3   ← model correctly knows this is highest ✅
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
# we will learn to transform the data on the train column
# and then test it on the test column
# we wont just transform the whole column


# ============================================================
# SAMPLE DATA
# ============================================================

data = {
    'education' : ['School', 'UG', 'PG', 'UG', 'PG',
                   'School', 'UG', 'PG', 'School', 'UG'],
    'review'    : ['Average', 'Good', 'Excellent', 'Good',
                   'Average', 'Poor', 'Average', 'Excellent',
                   'Good', 'Poor'],
    'purchased' : ['No', 'Yes', 'Yes', 'No', 'No',
                   'Yes', 'No', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# WE WILL JUST USE EDUCATION AND REVIEW AND PURCHASED COLUMN


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    df[["education", "review"]],
    df["purchased"],
    test_size=0.2
)


# ============================================================
# ORDINAL ENCODING
# ============================================================

# We make an object of OrdinalEncoder and pass the category
# order manually — lowest rank first, highest rank last.
# It will give the lowest values to School and Poor,
# and the highest values to PG and Excellent.

encoder = OrdinalEncoder(
    categories=[
        ["School", "UG", "PG"],               # education order
        ["Poor", "Average", "Good", "Excellent"]  # review order
    ]
)

# THE GOLDEN RULE — same as scaling:
# fit() on TRAIN only → learn the encoding from train data
# transform() on BOTH train and test

encoder.fit(X_train)

# Now transform both
X_train = encoder.transform(X_train)
X_test  = encoder.transform(X_test)

# After encoding:
# School=0, UG=1, PG=2
# Poor=0, Average=1, Good=2, Excellent=3

print(df['purchased'].unique())
# this will help you find all unique values in the dataset
# similarly you can do this for LabelEncoder too


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. LabelEncoder  → y column (target/output) ONLY
#    OrdinalEncoder → X columns (input features) ONLY
#
# 2. WHY NOT LabelEncoder for X?
#    It assigns alphabetical order — no control.
#    Excellent becomes 0, Poor becomes 2 → model thinks
#    Poor > Excellent. That's just wrong.
#
# 3. In OrdinalEncoder, ALWAYS define category order manually:
#    categories=[["Low", "Medium", "High"]]
#    Never let sklearn guess — it will guess wrong.
#
# 4. THE GOLDEN RULE holds here too:
#    fit() on X_train only → transform X_train and X_test
#    Same logic as StandardScaler / MinMaxScaler
#
# 5. INTERVIEW TRAP:
#    Q: "Can LabelEncoder be used on features?"
#    Right answer: "Technically yes, but it assigns arbitrary
#    alphabetical order which implies false ranking to the
#    model. Always use OrdinalEncoder with defined order."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → One Hot Encoding in depth
#    Handling nominal categories with no order,
#    the dummy variable trap, and drop='first' fix.
# ============================================================