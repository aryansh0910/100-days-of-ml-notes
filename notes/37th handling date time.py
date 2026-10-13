# ============================================================
# DATE TIME FEATURES — EXTRACTING USEFUL INFO FROM DATES
# ============================================================

# ============================================================
# WHAT IS IT?
# Date columns like "2025-01-01 08:30:00" look like simple
# strings to Python. But they contain HIDDEN FEATURES like:
# → What month is it? (January = holiday season?)
# → What day of week? (Monday vs Friday behaviour differs!)
# → Is it a weekend? (bike rentals drop on weekdays?)
# → What hour? (morning rush vs evening rush?)
#
# THE FIRST THING YOU SHOULD DO AFTER RECOGNIZING A
# DATETIME COLUMN IS USE pd.to_datetime() ON IT SO YOU
# CAN ACCESS ALL DATETIME FUNCTIONS ON THAT COLUMN.
#
# WHY?
# By default pandas reads date columns as STRINGS (object).
# String "2025-01-01" cannot do date math or extract parts.
# After pd.to_datetime() → it becomes a real datetime object
# → now you can extract year, month, day, hour etc. easily!
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import datetime


# ============================================================
# SAMPLE DATA — BIKE RENTALS AT A STATION
# ============================================================

data = {
    'timestamp': [
        '2025-01-01 08:30:00',
        '2025-01-01 17:45:00',
        '2025-01-02 09:15:00',
        '2025-01-04 12:00:00',
        '2025-01-05 21:30:00',
        '2025-02-14 18:00:00'
    ],
    'rentals'     : [15, 42, 12, 55, 8, 60],
    'temp_celsius': [4, 6, 2, 10, 3, 12]
}

df = pd.DataFrame(data)

print("BEFORE conversion:")
print(f"timestamp dtype: {df['timestamp'].dtype}")
# dtype = object (just a string, can't do date math on it!)
print(df['timestamp'])


# ============================================================
# STEP 1 — ALWAYS CONVERT TO DATETIME FIRST
# ============================================================

df['timestamp'] = pd.to_datetime(df['timestamp'])
# pd.to_datetime() converts string → actual datetime object
# NOW you can access all .dt properties on this column!

print("\nAFTER conversion:")
print(f"timestamp dtype: {df['timestamp'].dtype}")
# dtype = datetime64 ← now it's a real datetime! ✅


# ============================================================
# STEP 2 — EXTRACTING DATE PARTS
# ============================================================
#
# After converting to datetime, use .dt accessor to extract
# any part of the date. Think of .dt as "open the date and
# pull out what you need"

# Extract DATE only (without time)
df['date_only'] = df['timestamp'].dt.date
# THIS WOULD HELP YOU TO EXTRACT JUST THE DATE PART
# "2025-01-01 08:30:00" → "2025-01-01"

# Extract YEAR
df['year'] = df['timestamp'].dt.year
# SIMILARLY YOU CAN EXTRACT YEAR, MONTH AND OTHERS AS WELL
# "2025-01-01" → 2025

# Extract MONTH NUMBER
df['month'] = df['timestamp'].dt.month
# "2025-01-01" → 1 (January)

# Extract MONTH NAME
df['month_name'] = df['timestamp'].dt.month_name()
# "2025-01-01" → "January"
# Useful for plots and readable reports!

# Extract DAY NUMBER
df['day'] = df['timestamp'].dt.day
# "2025-01-01" → 1

# Extract TIME only (without date)
df['time_only'] = df['timestamp'].dt.time
# THIS WOULD GIVE US ONLY THE TIME PART
# "2025-01-01 08:30:00" → "08:30:00"

# Extract HOUR
df['hour'] = df['timestamp'].dt.hour
# "2025-01-01 08:30:00" → 8
# Very useful! Rentals at 8am vs 8pm behave very differently

print("\nExtracted date parts:")
print(df[['timestamp', 'date_only', 'year',
          'month', 'month_name', 'day',
          'time_only', 'hour']])


# ============================================================
# STEP 3 — DAY OF WEEK FEATURES
# ============================================================
#
# YOU CAN ALSO CHECK WHICH DAY OF THE WEEK IT WAS
# AND THE NAME OF THE DAY AS WELL

# Day of week as NUMBER (0=Monday, 6=Sunday)
df['date_dow'] = df['timestamp'].dt.dayofweek
# Monday = 0, Tuesday = 1 ... Sunday = 6
# Useful for models (numbers only, no strings)

# Day of week as NAME
df['date_dow_name'] = df['timestamp'].dt.day_name()
# Monday, Tuesday ... Sunday
# Useful for readable analysis and plots

# Is it a WEEKEND? (1=Yes, 0=No)
df['date_is_weekend'] = np.where(
    df['date_dow_name'].isin(['Saturday', 'Sunday']),
    1,   # if Saturday or Sunday → 1
    0    # otherwise → 0
)
# .isin(['Saturday', 'Sunday']) checks if value is in the list
# np.where() then converts True/False → 1/0

print("\nDay of week features:")
print(df[['timestamp', 'date_dow',
          'date_dow_name', 'date_is_weekend']])


# ============================================================
# STEP 4 — TIME ELAPSED BETWEEN DATES
# ============================================================
#
# Very useful feature! How many DAYS AGO was this record?
# Recent data might behave differently from old data.

today = datetime.datetime.today()
print(f"\nToday's date: {today}")

# Subtract the date column FROM today → gives timedelta object
# .dt.days → converts timedelta to NUMBER of days
df['days_ago'] = (today - df['timestamp']).dt.days
print("\nDays elapsed since each rental:")
print(df[['timestamp', 'days_ago']])

# WHY IS THIS USEFUL?
# A house listed for sale 300 days ago vs 2 days ago
# behaves very differently in price prediction!
# Time elapsed = a POWERFUL hidden feature in many datasets


# ============================================================
# STEP 5 — BONUS FEATURES YOU CAN EXTRACT
# ============================================================

# Quarter of year (Q1, Q2, Q3, Q4)
df['quarter'] = df['timestamp'].dt.quarter
# Jan-Mar=1, Apr-Jun=2, Jul-Sep=3, Oct-Dec=4

# Week number of the year
df['week_of_year'] = df['timestamp'].dt.isocalendar().week

# Is it morning, afternoon or evening?
df['time_of_day'] = pd.cut(
    df['hour'],
    bins=[0, 12, 17, 21, 24],
    labels=['Morning', 'Afternoon', 'Evening', 'Night'],
    right=False
)
# pd.cut() → manual binning based on hour value
# 0-12  → Morning
# 12-17 → Afternoon
# 17-21 → Evening
# 21-24 → Night

print("\nBonus features:")
print(df[['timestamp', 'quarter',
          'week_of_year', 'time_of_day']])


# ============================================================
# COMPLETE DATETIME FEATURE LIST — QUICK REFERENCE
# ============================================================
#
# ALWAYS START WITH:
# df['col'] = pd.to_datetime(df['col'])
#
# THEN EXTRACT WHATEVER YOU NEED:
#
# df['col'].dt.date          → date only (no time)
# df['col'].dt.time          → time only (no date)
# df['col'].dt.year          → year as integer
# df['col'].dt.month         → month as integer (1-12)
# df['col'].dt.month_name()  → month as string (January...)
# df['col'].dt.day           → day of month (1-31)
# df['col'].dt.hour          → hour (0-23)
# df['col'].dt.minute        → minute (0-59)
# df['col'].dt.dayofweek     → day of week (0=Mon, 6=Sun)
# df['col'].dt.day_name()    → day name (Monday...)
# df['col'].dt.quarter       → quarter (1,2,3,4)
# (today - df['col']).dt.days→ days elapsed since date


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. ALWAYS convert datetime strings to datetime objects FIRST
#    df['col'] = pd.to_datetime(df['col'])
#    Without this → it's just a string, can't extract anything!
#
# 2. Use .dt accessor to extract any date part
#    df['col'].dt.year, .dt.month, .dt.hour etc.
#
# 3. Day of week is a POWERFUL feature for many problems:
#    Monday behaviour ≠ Friday behaviour ≠ Sunday behaviour
#
# 4. Is Weekend feature (0/1) is very useful:
#    np.where(df['dow_name'].isin(['Saturday','Sunday']), 1, 0)
#
# 5. Time elapsed = (today - df['date']).dt.days
#    Recent records often behave very differently from old ones
#
# 6. Hour of day is crucial for time series problems
#    Morning rush, lunch, evening rush all differ significantly
# ============================================================


# ============================================================
# COMING UP NEXT:
# → ML Algorithms — Linear Regression
#    Preprocessing is now COMPLETELY done!
#    Time to finally train our first model and understand
#    how Linear Regression learns from data!
# ============================================================