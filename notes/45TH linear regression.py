# ============================================================
# LINEAR REGRESSION — FUNDAMENTALS
# ============================================================

# ============================================================
# WHAT IS IT?
#
# If PCA was about finding the best perspective to look at
# your data, Linear Regression is about finding the TREND
# LINE that best explains the relationship between your
# features and your target.
#
# Imagine you have a scatter plot of Annual Income (x) vs
# Spending Score (y). You want to draw a straight line
# through those points so that the line stays as close to
# all points as possible.
#
# The equation for this line is:
# y = w1*x1 + b
#
# (in ML terms, instead of y = mx + b)
#
# w1 (Weight/Slope) → how much y changes when x moves by 1 unit
# b  (Bias/Intercept) → where the line hits the y-axis when x=0
#
# Since your Churn dataset has a binary target (Churn vs
# No Churn), we usually use Logistic Regression for
# classification. But let's master the geometric and
# mathematical foundations of Linear Regression first,
# as everything else builds on this.
# ============================================================


# ============================================================
# 1. THE GEOMETRIC INTUITION — "THE BEST FIT LINE"
#
# THE COMPUTER DRAWS A BEST FIT LINE WHICH WILL CAUSE THE
# LEAST ERROR TO OCCUR. MEANS THAT LINE WHICH PASSES
# CLOSEST TO EVERY POINT IN THE DATASET.
#
# VISUAL:
#
#  Spending  |        ●
#  Score (y) |     ●     ╱●
#            |  ●     ╱     ●
#            |     ╱●           ← best fit line
#            |  ╱      ●
#            |________________________
#                Annual Income (x)
#
# The line doesn't pass through every point exactly —
# it minimizes the OVERALL distance from all points.
# ============================================================


# ============================================================
# 2. THE MATH — ORDINARY LEAST SQUARES (OLS)
#
# How does the computer "decide" which line is the best?
# It uses a cost function called Mean Squared Error (MSE).
#
# STEP BY STEP:
# 1. It draws a random line
# 2. It measures the vertical distance (the RESIDUAL)
#    between each actual data point and the line
# 3. It squares those distances (removes negative signs,
#    punishes big errors more)
# 4. It calculates the average of those squares
#
# THE GOAL: Find the w and b that make the total MSE as
# small as possible.
#
# VISUAL — Residuals:
#
#  y |        ● ← actual point
#    |       ╱│  ← residual (vertical distance)
#    |      ╱ │
#    |     ╱──┘  ← predicted point on the line
#    |    ╱
#    |___╱______________ x
#
# Why SQUARE the residuals instead of just adding them?
# → Removes negative signs (a residual of -5 and +5
#   shouldn't cancel out to 0 error)
# → Punishes large errors MORE than small ones
#   (an error of 10 becomes 100, an error of 2 becomes 4 —
#   the model is forced to avoid big mistakes)
# ============================================================


# ============================================================
# 4. ASSUMPTIONS OF LINEAR REGRESSION
#
# Linear Regression is a bit "picky." For it to work well,
# your data should follow these rules:
#
# 1. LINEARITY:
#    The relationship between x and y must be a straight
#    line (not a curve).
#
# 2. NO MULTICOLLINEARITY:
#    Your features shouldn't be too highly correlated with
#    each other (this is why we checked for the Curse of
#    Dimensionality!).
#
# 3. HOMOSCEDASTICITY:
#    The "spread" of your errors should be constant —
#    not bigger for some x values and smaller for others.
#
# 4. NORMALITY OF RESIDUALS:
#    The errors should follow a normal distribution.
#
# VISUAL — Homoscedasticity vs Heteroscedasticity:
#
# HOMOSCEDASTIC (good, constant spread):
#  error |  ●  ●     ●    ●  ●
#        |●  ●   ●  ●  ●    ●
#        |__________________________ x
#        (errors look roughly the same width everywhere)
#
# HETEROSCEDASTIC (bad, violates assumption):
#  error |              ●
#        |          ● ●   ●
#        |  ● ●   ●    ●    ●
#        |●_________________________ x
#        (errors get WIDER as x increases — pattern in errors
#         means model isn't capturing something)
# ============================================================


# ============================================================
# CODE — BASIC LINEAR REGRESSION
# ============================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Sample data
np.random.seed(42)
income  = np.random.randint(20000, 150000, 100)
spending = 0.0003 * income + np.random.normal(0, 5, 100)

df = pd.DataFrame({'Income': income, 'Spending_Score': spending})

X = df[['Income']]
y = df['Spending_Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Weight (slope):", model.coef_)
print("Bias (intercept):", model.intercept_)

# Predicting
predictions = model.predict(X_test)


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Equation: y = w1*x1 + b
#    w = weight/slope, b = bias/intercept
#
# 2. OLS finds the best w, b by MINIMIZING the sum of
#    squared residuals (errors)
#
# 3. Squaring residuals: removes negative signs AND
#    punishes large errors more than small ones
#
# 4. Four assumptions: Linearity, No Multicollinearity,
#    Homoscedasticity, Normal Residuals
#    If these are violated, Linear Regression performs poorly
#
# 5. For binary classification (like Churn), use Logistic
#    Regression instead — Linear Regression is for continuous
#    targets
#
# 6. INTERVIEW TRAP:
#    Q: "Why do we square the residuals instead of taking
#        absolute value?"
#    Right answer: "Squaring is differentiable everywhere,
#    which makes gradient descent optimization possible.
#    Absolute value has a sharp corner at zero where the
#    derivative doesn't exist cleanly. Squaring also
#    penalizes large errors more heavily."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Logistic Regression
#    Adapting the linear equation with a Sigmoid function
#    to handle binary classification problems like Churn.
# ============================================================