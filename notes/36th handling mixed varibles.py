# ============================================================
# MIXED VARIABLES — HANDLING COLUMNS WITH NUMBERS + TEXT
# ============================================================

# ============================================================
# WHAT IS IT?
# Mixed Variables are like a "buy one, get one free" deal
# that you didn't ask for. They are columns that contain
# BOTH numbers AND text (strings) in the SAME cell.
#
# MOST COMMON IS LIKE A SAMPLE COLUMN THAT INCLUDES
# NUMBERS AS WELL AS CHARACTERS LIKE A, B, 1, 4
#
# Common examples:
# Ticket Numbers : PC 17599, 7534, A/5 21171
# Cabin Numbers  : C23, B51, E46
# Age column     : 22, 38, "Z", "A" (data entry errors)
#
# We CAN'T just scale or encode them directly.
# We have to SEPARATE them first into two clean columns!
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# TYPES OF MIXED VARIABLES
# ============================================================
#
# TYPE 1 → Numbers + Strings in SAME column
#   Example: age column has [22, 38, "Z", "A"]
#   Some rows have real ages, some have typos/errors
#
# TYPE 2 → Numbers + Strings WITHIN SAME CELL
#   Example: ticket column has ["PC 17599", "A/5 21171", "7534"]
#   One cell contains BOTH a prefix AND a number together
#
# TWO DIFFERENT PROBLEMS → TWO DIFFERENT SOLUTIONS!


# ============================================================
# SAMPLE DATA
# ============================================================

data = {
    'ticket': ['A/5 21171', 'PC 17599', '113803',
               'PP 9549', 'A/4 48871', '232912'],
    'cabin' : ['C123', 'B45', np.nan, 'E121', 'D', 'F5'],
    'age'   : [22, 38, 26, 35, "Z", "A"]
}

df = pd.DataFrame(data)
print("ORIGINAL DATAFRAME:")
print(df)
print(f"\nAge column dtype: {df['age'].dtype}")
# dtype = object because it has MIXED types (int + string)


# ============================================================
# SOLUTION 1 — TYPE 1 MIXED (Numbers + Strings in same column)
# ============================================================
#
# GOAL: Split age column into:
# age_numerical   → [22, 38, 26, 35, NaN, NaN]  (real numbers)
# age_categorical → [NaN, NaN, NaN, NaN, "Z", "A"] (text errors)
#
# TO HANDLE THE NUMERICAL COLUMN WITH CHARACTERS IN IT
# WE HAVE TO SEPARATE THOSE COLUMNS INTO TWO PARTS AND
# REPLACE THE PARTS WITH NaN VALUES IN PLACE OF EACH OTHER

# ── STEP 1: Extract NUMERICAL part ───────────────────────────
df['age_numerical'] = pd.to_numeric(df['age'],
                                     errors='coerce',
                                     downcast='integer')
# pd.to_numeric() → tries to convert everything to a number
#
# errors='coerce' → THE MOST IMPORTANT PARAMETER!
# If pandas sees something it CAN'T convert to number
# (like "Z" or "A"), instead of CRASHING it converts
# that value to NaN (Not a Number)
# "coerce" literally means "force it, use NaN if you can't"
#
# downcast='integer' → stores as integer not float
#                      (saves memory)
#
# RESULT:
# 22  → 22   ✅ (real number, kept)
# "Z" → NaN  ✅ (text, replaced with NaN)
# "A" → NaN  ✅ (text, replaced with NaN)

print("\nAge numerical column:")
print(df['age_numerical'])


# ── STEP 2: Extract CATEGORICAL part ─────────────────────────
df['age_categorical'] = np.where(
    df['age_numerical'].isnull(),  # CONDITION: is it NaN?
    df['age'],                     # IF TRUE: grab original value
    np.nan                         # IF FALSE: fill with NaN
)
# np.where(condition, value_if_true, value_if_false)
#
# THIS MEANS WHERE df['age_numerical'] HAS A NULL VALUE
# THAT WOULD BE FILLED WITH THE ORIGINAL VALUES OF THE
# AGE COLUMN AND THE OTHER VALUES WHERE isnull() IS FALSE
# FILL WITH np.nan VALUES
#
# LOGIC:
# If age_numerical IS null → original was TEXT → grab original
# If age_numerical is NOT null → original was NUMBER → NaN
#
# RESULT:
# Row where age=22  → age_numerical=22,  age_categorical=NaN
# Row where age="Z" → age_numerical=NaN, age_categorical="Z"

print("\nAge categorical column:")
print(df['age_categorical'])

print("\nAfter separation:")
print(df[['age', 'age_numerical', 'age_categorical']])

# AFTER SEPARATION — fill NaN values:
# age_numerical   → fill NaN with MEDIAN (most logical for age)
# age_categorical → fill NaN with MOST FREQUENT value or "Unknown"
df['age_numerical']   = df['age_numerical'].fillna(
                        df['age_numerical'].median())
df['age_categorical'] = df['age_categorical'].fillna('Unknown')

print("\nAfter filling NaN values:")
print(df[['age_numerical', 'age_categorical']])


# ============================================================
# SOLUTION 2 — TYPE 2 MIXED (Numbers + Text IN SAME CELL)
# ============================================================
#
# PROBLEM: Ticket column has ["PC 17599", "A/5 21171", "113803"]
# One cell has BOTH a text prefix AND a number inside it.
#
# GOAL: Split into:
# ticket_prefix → ["PC", "A/5", NaN]      (text part)
# ticket_number → [17599, 21171, 113803]   (number part)

print("\n" + "=" * 55)
print("TYPE 2 MIXED — Text + Number IN SAME CELL")
print("=" * 55)
print("Ticket column:")
print(df['ticket'])

# ── STEP 1: Extract NUMERIC part from ticket ──────────────────
# regex r'\d+' means "find sequences of DIGITS"
# r'\d+' → \d = any digit (0-9), + = one or more
df['ticket_number'] = df['ticket'].str.extract(r'(\d+)')
# .str.extract() → applies regex to EACH cell
# captures the FIRST group of digits found in the string
# "PC 17599"  → extracts "17599"
# "A/5 21171" → extracts "5" (FIRST digits found!) ⚠️
# "113803"    → extracts "113803"

# Convert extracted number to actual integer
df['ticket_number'] = pd.to_numeric(df['ticket_number'],
                                     errors='coerce')

# ── STEP 2: Extract TEXT/PREFIX part from ticket ──────────────
# regex r'([a-zA-Z/]+)' means "find sequences of LETTERS"
# [a-zA-Z/] = any letter (uppercase or lowercase) or /
df['ticket_prefix'] = df['ticket'].str.extract(r'([a-zA-Z/]+)')
# "PC 17599"  → extracts "PC"
# "A/5 21171" → extracts "A/"
# "113803"    → NaN (no letters found)

print("\nAfter splitting ticket into prefix + number:")
print(df[['ticket', 'ticket_prefix', 'ticket_number']])


# ============================================================
# CABIN COLUMN — BONUS EXAMPLE (most common interview question)
# ============================================================
#
# Cabin column: ['C123', 'B45', NaN, 'E121', 'D', 'F5']
# C123 = Deck C + Cabin Number 123
# We can extract the DECK LETTER separately!

print("\n" + "=" * 55)
print("CABIN — Extract Deck Letter + Cabin Number")
print("=" * 55)

# Extract deck letter (first character)
df['cabin_deck'] = df['cabin'].str[0]
# .str[0] → gets the FIRST character of each string
# 'C123' → 'C', 'B45' → 'B', NaN → NaN

# Extract cabin number (digits only)
df['cabin_number'] = df['cabin'].str.extract(r'(\d+)')
df['cabin_number'] = pd.to_numeric(df['cabin_number'],
                                    errors='coerce')

print(df[['cabin', 'cabin_deck', 'cabin_number']])


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. TWO types of mixed variables:
#    Type 1 → numbers AND strings in SAME column (different rows)
#             Example: age=[22, 38, "Z", "A"]
#    Type 2 → numbers AND strings in SAME CELL
#             Example: ticket=["PC 17599", "113803"]
#
# 2. For TYPE 1 → use pd.to_numeric(errors='coerce')
#    → converts numbers, turns text into NaN automatically
#    → then use np.where() to extract the text part separately
#
# 3. errors='coerce' is the KEY parameter
#    Without it → pd.to_numeric() CRASHES on text values
#    With it    → safely converts text to NaN ✅
#
# 4. np.where(condition, value_if_true, value_if_false)
#    → like an IF-ELSE but applied to entire column at once
#
# 5. For TYPE 2 → use regex with .str.extract()
#    r'\d+'        → extracts digits (numbers)
#    r'[a-zA-Z/]+' → extracts letters (text prefix)
#
# 6. After separating → always fill NaN values:
#    Numerical part   → fill with median
#    Categorical part → fill with mode or "Unknown"
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Date and Time Features
#    Extracting day, month, year, day of week from
#    datetime columns — another very common real world
#    data engineering task!
# ============================================================