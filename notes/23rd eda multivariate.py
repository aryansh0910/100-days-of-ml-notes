# ============================================================
# BIVARIATE & MULTIVARIATE ANALYSIS — EDA LEVEL 2 & 3
# ============================================================

# ============================================================
# WHAT IS IT?
# Bivariate   = looking at TWO variables together
# Multivariate = looking at THREE or more variables together
#
# It's where you stop looking at one or two things and
# start looking at the WHOLE PICTURE at once —
# 3, 4, or even 10 variables together.
#
# PROGRESSION OF EDA:
# Univariate   → "How many people died on Titanic?"
# Bivariate    → "Did men die more than women?"
# Multivariate → "Did a RICH MAN in 3rd class have a better
#                 chance than a POOR WOMAN in 3rd class?"
#                 (Age + Sex + Class + Survived together)
#
# This is where the REAL insights come from.
# Univariate gives you facts.
# Multivariate gives you STORIES.
# ============================================================


# ============================================================
# IMPORTS AND DATA LOADING
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading multiple datasets for different examples
tips    = sns.load_dataset("tips")      # restaurant tips data
flights = sns.load_dataset("flights")   # flight passenger data
iris    = sns.load_dataset("iris")      # famous flower dataset

df = pd.read_csv("22nd_train_data.csv") # Titanic dataset

# Reconstruct Pclass from one-hot encoded columns
# WE MULTIPLIED THIS BECAUSE IT WAS ORIGINALLY MADE USING
# pd.get_dummies() FUNCTION SO IT WOULD JUST HAVE A SINGLE
# 1 VALUE AND REST OTHERS WOULD BE 0.
# SO WE MULTIPLIED AND ADDED THEM SO THEY LOOK LIKE THIS:
# IN CASE OF PCLASS 2 → 0×1 + 1×2 + 0×3 = 2
# SO THERE WOULD BE 3 CATEGORIES NAMED AS 1, 2, 3
df["Pclass"] = df["Pclass_1"]*1 + df["Pclass_2"]*2 + df["Pclass_3"]*3


# ============================================================
# DECISION GUIDE — WHICH PLOT FOR WHICH COMBINATION?
# ============================================================
#
# ┌──────────────────────┬──────────────────────────────────┐
# │ Data Combination     │ Best Plot                        │
# ├──────────────────────┼──────────────────────────────────┤
# │ Numerical + Numerical│ Scatter Plot                     │
# │ Numerical + Category │ Bar Plot / Box Plot              │
# │ Category + Category  │ Countplot with hue               │
# │ All Numericals       │ Pair Plot / Heatmap              │
# │ 3+ variables         │ Scatter with hue + style         │
# └──────────────────────┴──────────────────────────────────┘
# ============================================================


plt.figure(figsize=(18, 14))


# ============================================================
# PLOT 1 — SCATTER PLOT (Numerical vs Numerical) BIVARIATE
# ============================================================
#
# A scatter plot tells you "how one variable REACTS
# when the other CHANGES."
#
# WHAT TO LOOK FOR:
# → Points going UP-RIGHT   = Positive correlation
#   (as total_bill increases, tip increases)
# → Points going DOWN-RIGHT = Negative correlation
# → Random spread           = No correlation
#
# WHY IT MATTERS FOR ML:
# When you start Linear Regression, you need to know
# if a STRAIGHT LINE can actually represent your data.
# If the dots form a CURVE → simple linear model won't work
# You might need Polynomial Regression instead.
# The scatter plot is your "sanity check" BEFORE running
# any math or fitting any model!

plt.subplot(3, 3, 1)
plt.scatter(tips["total_bill"], tips["tip"],
            alpha=0.6, color="steelblue", edgecolors="black")
plt.title("Total Bill vs Tip (Basic Scatter)")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip ($)")
# HOW TO READ THIS:
# Points clustered bottom-left → most bills and tips are low
# Upward trend → higher bill = higher tip (positive correlation)
# Wide spread  → relationship exists but not perfectly linear


# ============================================================
# PLOT 2 — SCATTER PLOT WITH HUE + STYLE (MULTIVARIATE)
# ============================================================
#
# NOW SIMILARLY WE CAN USE THIS SAME PLOT FOR MULTIVARIATE
# ANALYSIS AS WELL BY ADDING ANOTHER COLUMN AS HUE.
# IT WILL CHANGE THE COLOR ACCORDING TO THE CATEGORIES.
# LIKE IF WE WANT TO KNOW IF THE PERSON IS FEMALE OR MALE
# WHO PAID THE TIPS WE CAN USE THE HUE PARAMETER.
# BUT THAT FEATURE IS IN SEABORN SO WE HAVE TO USE sns.
#
# WE CAN ALSO ADD ANOTHER PARAMETER KNOWN AS STYLE.
# IT WILL JUST CHANGE THE STYLE OF MARKING THE POINT
# LIKE USE O FOR YES AND X FOR NO.
# SO WE WILL TRY TO FIND THE SMOKERS.
#
# hue   → changes COLOR by category (Sex: Male/Female)
# style → changes MARKER SHAPE by category (Smoker: Yes/No)
#
# Now ONE plot is telling us FOUR things at once:
# total_bill, tip, sex, AND smoker status!
# This is the power of Multivariate Analysis.

plt.subplot(3, 3, 2)
sns.scatterplot(x=tips["total_bill"],
                y=tips["tip"],
                hue=tips["sex"],       # color by gender
                style=tips["smoker"])  # shape by smoker status
plt.title("Total Bill vs Tip\n(hue=Sex, style=Smoker)")
plt.xlabel("Total Bill ($)")
plt.ylabel("Tip ($)")
# HOW TO READ THIS:
# Each color = one gender (Male/Female)
# Each shape = smoker or non-smoker
# Look for clusters → do male smokers tip differently?
# This is 4 variables in ONE single plot! 🔥


# ============================================================
# PLOT 3 — BAR PLOT (Numerical vs Categorical) BIVARIATE
# ============================================================
#
# NOW WE WILL SEE HOW TO PLOT IN CASE OF ONE NUMERICAL
# AND ONE CATEGORICAL COLUMN — WE DO THIS WITH BAR PLOT.
# WE WILL FIND THE AVERAGE FARE OF PERSON IN EACH PCLASS.
#
# IMPORTANT DISTINCTION:
# Histogram → continuous range of numbers (Age: 0 to 100)
# Bar Plot  → discrete groups (Pclass: 1st, 2nd, 3rd class)
# They LOOK similar but represent completely different things!
#
# seaborn barplot AUTOMATICALLY calculates the MEAN
# of the y variable for each category on x axis.
# The black line on each bar = CONFIDENCE INTERVAL
# (how certain we are about that mean value)
#
# U CAN ALSO USE HUE HERE FOR MULTIVARIATE ANALYSIS!
# Adding hue=Sex splits each Pclass bar by gender.
# Now we see: average fare by class AND by gender!

plt.subplot(3, 3, 3)
sns.barplot(x=df["Pclass"],
            y=df["Fare"],
            hue=df["Sex"])
plt.title("Average Fare by Pclass\n(hue=Sex) — Multivariate")
plt.xlabel("Passenger Class")
plt.ylabel("Average Fare (£)")
# HOW TO READ THIS:
# Taller bar    = higher average fare for that class
# Color split   = compare male vs female fare within same class
# 1st class bar much taller → confirms 1st class paid more
# Gender difference in fare? → check if bars differ by color


# ============================================================
# PLOT 4 — PAIRPLOT (All Numerical Columns Together)
# ============================================================
#
# U CAN USE PAIRPLOT — IT HELPS YOU PLOT THE SCATTER PLOT
# BETWEEN ALL NUMERICAL COLUMNS IN THE DATA WITH EACH OTHER.
# U CAN ALSO USE HUE IN THIS TO SEE CATEGORY DIFFERENTIATION.
#
# What pairplot gives you:
# → Diagonal    = distribution of each variable (KDE/histogram)
# → Off-diagonal = scatter plot between every pair of columns
# → hue         = color each species differently
#
# For IRIS dataset with 4 features:
# pairplot creates a 4×4 grid = 16 plots automatically!
# This gives you a COMPLETE picture of all relationships
# in your data in just ONE line of code.
#
# This is one of the MOST POWERFUL EDA tools available.
# Run it on every new dataset you get.

plt.subplot(3, 3, 4)
plt.text(0.5, 0.5,
         "Pairplot shown\nin separate window\n(sns.pairplot)",
         ha='center', va='center',
         fontsize=12, color='steelblue')
plt.axis('off')
plt.title("Pairplot → See Below")

# Pairplot opens in its own window since it creates
# a full grid — cannot be placed inside subplot
sns.pairplot(iris, hue="species",
             plot_kws={'alpha': 0.6},
             diag_kind='kde')
plt.suptitle("Iris Pairplot — All Features vs All Features",
             y=1.02, fontsize=14)

# HOW TO READ PAIRPLOT:
# Diagonal plots  → distribution of each single feature
# Off-diagonal    → scatter between two features
# Well separated colored clusters → features can distinguish
#                                   species well = good for ML!
# Overlapping clusters → harder to classify


# ============================================================
# BONUS — HEATMAP (Best for Correlation Overview)
# ============================================================
#
# While pairplot shows individual scatter plots,
# a HEATMAP shows ALL correlations as a single color grid.
# Faster to read when you have many features.
#
# Color scale:
# Dark Red  → strong positive correlation (+1)
# White     → no correlation (0)
# Dark Blue → strong negative correlation (-1)

plt.subplot(3, 3, 5)
numerical_cols = iris.drop("species", axis=1)
corr_matrix    = numerical_cols.corr()
sns.heatmap(corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            linewidths=0.5,
            square=True)
plt.title("Iris — Correlation Heatmap")
# HOW TO READ THIS:
# petal_length vs petal_width = 0.96 → almost perfect correlation
# Can probably drop one of them for the model!


# ============================================================
# BONUS — BOX PLOT BY CATEGORY (Numerical vs Categorical)
# ============================================================
#
# While bar plot shows the MEAN, box plot shows the
# FULL DISTRIBUTION of a numerical column per category.
# Much more information in one plot!

plt.subplot(3, 3, 6)
sns.boxplot(x=df["Sex"],
            y=df["Fare"],
            hue=df["Survived"] if "Survived" in df.columns
            else None)
plt.title("Fare Distribution by Sex\n(hue=Survived)")
plt.xlabel("Sex")
plt.ylabel("Fare (£)")
# HOW TO READ THIS:
# Box height    → spread of fares (IQR)
# Middle line   → median fare
# Dots outside  → outliers (very expensive tickets)
# Compare boxes → do men and women pay different fares?


plt.tight_layout()
plt.savefig('bivariate_multivariate_eda.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Bivariate & Multivariate EDA plots saved!\n")


# ============================================================
# THE POWER OF MULTIVARIATE THINKING
# ============================================================
#
# Titanic survival example — how analysis deepens:
#
# UNIVARIATE:
# "38% of passengers survived"
# → Just a number, not very useful
#
# BIVARIATE:
# "Women survived more than men" (Sex vs Survived)
# → Interesting! But still incomplete
#
# MULTIVARIATE:
# "A rich woman in 1st class had 95%+ survival chance
#  A poor man in 3rd class had <15% survival chance"
# (Age + Sex + Pclass + Fare + Survived all together)
# → NOW you have a real story and real insight!
#
# This is why multivariate analysis is the final boss
# of EDA — it reveals the COMPLETE truth in the data.
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Bivariate   = 2 variables, Multivariate = 3+ variables
#    Always progress from Uni → Bi → Multi for full picture
#
# 2. Scatter Plot = Numerical vs Numerical
#    Use it as "sanity check" before Linear Regression
#    Straight line pattern → Linear Regression will work ✅
#    Curved pattern        → Need Polynomial Regression ⚠️
#
# 3. hue in seaborn = add a 3rd categorical variable
#    style in seaborn = add a 4th categorical variable
#    One scatter plot can show 4 variables at once!
#
# 4. Bar Plot = Numerical vs Categorical
#    seaborn automatically calculates the MEAN
#    Black line on bar = confidence interval
#    Histogram ≠ Bar Plot (different purposes!)
#
# 5. Pairplot = scatter plot between ALL numerical columns
#    Most powerful single EDA command
#    Run it on every new dataset → one line of code!
#
# 6. Heatmap = fastest way to see ALL correlations at once
#    Value close to 1 or -1 → strong relationship
#    Value close to 0        → weak/no relationship
#    Two features 99% correlated → keep only ONE
#
# 7. Pclass was reconstructed from get_dummies columns:
#    Pclass = Pclass_1×1 + Pclass_2×2 + Pclass_3×3
#    This reverses one-hot encoding back to original column
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Feature Engineering
#    Creating new meaningful columns from existing ones.
#    Using the insights from EDA to build better features
#    that help the model learn patterns more effectively.
#    Example: from EDA we saw Sex matters a lot →
#    we encode it properly for the model.
# ============================================================ 