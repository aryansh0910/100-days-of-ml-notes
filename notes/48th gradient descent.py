# ============================================================
# GRADIENT DESCENT
# ============================================================

# ============================================================
# WHAT IS IT?
#
# This is the "engine" under the hood of almost every
# Machine Learning algorithm. If Linear Regression is the
# goal (finding the best line), Gradient Descent is the
# PROCESS of getting there.
#
# GRADIENT IS LIKE AN ALGO WHICH FINDS THE LOWEST LOSS
# FUNCTION FOR AN ALGO, OR U CAN SAY THE POINT WHERE THE
# ERRORS ARE THE LEAST FOR THE PREDICTED VALUES TO THE
# ACTUAL VALUES.
#
# Imagine you are standing at the top of a foggy mountain
# (a high error rate) and you want to reach the valley
# (the lowest possible error). You can't see the whole
# path, so you feel the slope under your feet and take a
# step in the direction that goes down.
# ============================================================


# ============================================================
# THE CORE LOGIC
#
# U TAKE A RANDOM VALUE OF A NUMBER ON THE GRAPH. WE KNOW
# THAT IT WILL BE A PARABOLIC GRAPH DUE TO ITS FORMULA:
# E = (Yi - y_pred)²  AND  y_pred = mx + b
#
# SO WE CHOOSE A RANDOM VALUE OF B IN THAT PARABOLIC GRAPH
# AND CHECK THE SLOPE OF THE GRAPH AT THAT POINT.
#
# IF THE SLOPE IS NEGATIVE → WE MOVE OUR VALUE TO THE RIGHT
# SO THAT THE SLOPE BECOMES ZERO AND WE FIND THE LOWEST
# ERROR VALUE.
#
# IF THE SLOPE IS POSITIVE → WE MOVE TO THE LEFT OF THE
# GRAPH TO CHOOSE THE B VALUES.
#
# FORMULA:
# ┌─────────────────────────────────────────────┐
# │   B_new  =  B_old  -  η × slope              │
# └─────────────────────────────────────────────┘
#
# η (Eta) → learning rate, the weightage each slope will
#           carry in the improvement
# slope   → slope at that point
#
# WE KEEP DOING THIS UNTIL WE FIND THE DIFFERENCE BETWEEN
# THE SLOPES OF TWO POINTS LESS THAN 0.001, OR WE SET THE
# ITERATIONS = 100, 1000 ASSUMING WE WILL FIND THE ANSWER
# IN THAT MANY ITERATIONS (ALSO KNOWN AS EPOCH).
#
# VISUAL — the parabolic cost function (U-shaped valley):
#
#  Error
#    |  ●
#    |    ●
#    |       ●
#    |   slope        ●
#    |  negative          ●          slope
#    |  here (move right)    ●       positive here
#    |                          ●    (move left)
#    |________________●____________●________
#                  minimum (target)
#                B value →
#
# At the minimum point, the slope = 0 (flat).
# That's the B value that gives the LOWEST error.
# ============================================================


# ============================================================
# 1. THE COST FUNCTION (THE VALLEY)
#
# To find the best line, we need to minimize the Mean
# Squared Error (MSE). In Gradient Descent, we treat MSE
# as a function of our weights (w) and bias (b). This
# creates a U-shaped curve called the Cost Function.
# ============================================================


# ============================================================
# 2. HOW IT WORKS — STEP BY STEP
#
# 1. RANDOM START:
#    The model starts with random values for w and b.
#    Usually, the error is very high here.
#
# 2. CALCULATE THE GRADIENT:
#    The "Gradient" is just a fancy word for the slope
#    (the derivative).
#    → If the slope is positive, move left
#    → If the slope is negative, move right
#
# 3. THE UPDATE RULE:
#    w_new = w_old - (η × gradient)
#    η (Eta) = the Learning Rate
# ============================================================


# ============================================================
# 3. THE LEARNING RATE (η)
#
# This is the most important hyperparameter in Gradient
# Descent. It controls the SIZE of the steps you take.
#
# TOO SMALL:
# You take tiny baby steps. The model will eventually
# reach the bottom, but it might take hours or days
# (Slow Convergence).
#
# TOO LARGE:
# You take giant leaps. You might OVERSTEP the valley and
# bounce back and forth, never actually reaching the bottom
# (Divergence).
#
# LEARNING RATE IS A VERY IMPORTANT PARAMETER — IF IT'S
# TOO HIGH U WILL SKIP THE SOLUTION, AND IF IT'S TOO LOW
# U WILL NEVER REACH THE SOLUTION.
#
# VISUAL:
#
# Too small η:
#  ●→●→●→●→●→●→●→●→●→●→●→●→●→●→● (takes forever to reach bottom)
#       very slow valley descent
#
# Just right η:
#  ●────→●──→●─→●→●●  (reaches bottom efficiently)
#
# Too large η:
#  ●←──────────────→●  ●←──→●
#  bouncing wildly across the valley, never settling ❌
# ============================================================


# ============================================================
# 4. TYPES OF GRADIENT DESCENT
#
# In your Churn project, depending on how much data you
# have, you might use different "flavors":
#
# ┌──────────────┬───────────────────┬─────────────┬──────────────┐
# │ Type         │ How it works      │ Pro         │ Con          │
# ├──────────────┼───────────────────┼─────────────┼──────────────┤
# │ Batch GD     │ Uses all rows to  │ Very stable │ Very slow    │
# │              │ take one step     │             │ for big data │
# │ Stochastic   │ Uses one random   │ Super fast; │ Very "jumpy" │
# │ (SGD)        │ row per step      │ escapes     │ and noisy    │
# │              │                   │ local minima│              │
# │ Mini-Batch GD│ Uses a small      │ Best of both│ Requires     │
# │              │ group (e.g. 32)   │ worlds      │ tuning batch │
# │              │                   │             │ size         │
# └──────────────┴───────────────────┴─────────────┴──────────────┘
#
# VISUAL — path each type takes to reach the minimum:
#
# Batch GD:    ●──→●──→●──→●──→●        (smooth, direct path)
#
# SGD:         ●─↗●↘──→●─↗●↘→●↗●→●     (jumpy, zigzag path)
#
# Mini-Batch:  ●─↗●─→●↘●─→●─→●          (mostly smooth, some noise)
# ============================================================


# ============================================================
# 5. WHY DO WE NEED THIS?
#
# Why not just use a mathematical formula to find the
# bottom directly?
#
# For simple linear regression, we CAN — it's called the
# Normal Equation. But when you have millions of rows or
# thousands of features (The Curse of Dimensionality!),
# the math becomes too heavy for a computer to solve at
# once.
#
# Gradient Descent allows us to find the answer ITERATIVELY,
# one small step at a time — much more computationally
# feasible at scale.
# ============================================================


# ============================================================
# CODE — SIMPLE GRADIENT DESCENT FROM SCRATCH
# ============================================================

import numpy as np

# Sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3 * X.flatten() + 5 + np.random.randn(100) * 2

# Initialize random weight and bias
w = 0.0
b = 0.0
learning_rate = 0.01
epochs = 1000
n = len(X)

for epoch in range(epochs):
    y_pred = w * X.flatten() + b

    # Calculate gradients (derivatives of MSE wrt w and b)
    dw = (-2/n) * np.sum(X.flatten() * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)

    # Update rule
    w = w - learning_rate * dw
    b = b - learning_rate * db

print(f"Final w: {w:.4f}, Final b: {b:.4f}")
# Should converge close to w=3, b=5 (the true relationship)


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Gradient Descent = iterative optimization to find
#    minimum error (lowest point on the cost function curve)
#
# 2. Update rule: w_new = w_old - (η × gradient)
#    Negative slope → move right, Positive slope → move left
#
# 3. Learning rate (η) is the MOST important hyperparameter:
#    Too small → slow convergence
#    Too large → divergence (never settles, bounces around)
#
# 4. Three types:
#    Batch GD → all data, stable, slow
#    SGD → one row, fast, noisy
#    Mini-Batch GD → small batches, balanced (most used)
#
# 5. Why not solve directly with Normal Equation?
#    Works for small data, but becomes computationally
#    impossible for millions of rows or thousands of features
#
# 6. INTERVIEW TRAP:
#    Q: "What happens if learning rate is too high?"
#    Wrong: "Model trains faster"
#    Right: "Model may overshoot the minimum and DIVERGE —
#    bouncing back and forth across the valley without ever
#    converging, sometimes even increasing the error over
#    iterations instead of decreasing it."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Logistic Regression
#    Adapting the linear equation with a Sigmoid function
#    to handle binary classification problems like Churn.
# ============================================================