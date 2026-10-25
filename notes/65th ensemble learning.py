# ============================================================
# ENSEMBLE LEARNING
# ============================================================

# ============================================================
# WHAT IS IT?
#
# Ensemble learning is where machine learning gets really
# powerful. If a single Decision Tree is like asking ONE
# PERSON for their opinion, Ensemble Learning is like
# asking a whole "WISDOM OF THE CROWD."
#
# WISDOM OF THE CROWD: The idea is that while one person
# might be wrong, a large group of people can often get
# the right answer by averaging out their opinions. In ML,
# we do this by combining multiple models to make a final
# prediction.
# ============================================================


# ============================================================
# TWO WAYS TO BUILD AN ENSEMBLE
#
# BY DIFFERENT MODELS:
# You could combine a Decision Tree with a Logistic
# Regression model. Each model might capture different
# patterns in the data, and when combined, they can provide
# a more accurate prediction than either model alone.
#
# OR
#
# U CAN KEEP THE ALGO SAME AND GIVE THEM DIFFERENT DATA:
# For example, you could train 100 Decision Trees, but
# each tree sees a different random sample of the data
# (this is called "Bagging").
#
# SO WHILE PREDICTING WE WILL GIVE THE DATA TO BE PREDICTED
# TO ALL THE MODELS AND THEN:
# → Average their predictions (FOR REGRESSION)
# → Take a vote (FOR CLASSIFICATION)
# ============================================================


# ============================================================
# TYPES OF ENSEMBLE METHODS
#
# 0. VOTING (SIMPLEST FORM):
#    ALL ALGOS ARE SAME OR DIFFERENT, BUT THEY ARE TRAINED
#    INDEPENDENTLY, AND THEN WE TAKE A VOTE OR AVERAGE.
#    WE CAN USE THE SAME MODEL WITH DIFFERENT
#    HYPERPARAMETERS TO GET BETTER RESULTS.
#
# 1. BAGGING (Bootstrap Aggregating):
#    Random Forest (includes a lot of Decision Trees) is
#    the most famous example.
#    IT MEANS USING SAME MODELS JUST GIVING THEM DIFFERENT
#    DATA, SO EACH TREE SEES A DIFFERENT RANDOM SAMPLE OF
#    THE DATA, AND THEN WE AVERAGE THEIR PREDICTIONS.
#
# 2. BOOSTING:
#    AdaBoost, Gradient Boosting, and XGBoost are popular
#    examples.
#    IN THIS APPROACH, WE TRAIN MODELS SEQUENTIALLY, WHERE
#    EACH MODEL TRIES TO FIX THE MISTAKES OF THE PREVIOUS
#    ONE. THE FINAL PREDICTION IS A WEIGHTED COMBINATION
#    OF ALL MODELS.
#
# 3. STACKING:
#    ITS LIKE VOTING ONLY BUT U JUST ASSIGN THE WEIGHTS TO
#    THE MODELS BASED ON THEIR PERFORMANCE, SO THE BETTER
#    MODEL GETS MORE WEIGHT IN THE FINAL PREDICTION.
#    (WE ANALYSE THE PERFORMANCE OF EACH MODEL ON A
#    VALIDATION SET AND THEN ASSIGN WEIGHTS ACCORDINGLY.
#    THE BETTER THE MODEL PERFORMS, THE MORE INFLUENCE IT
#    HAS ON THE FINAL PREDICTION.)
#
# VISUAL — all four types side by side:
#
# VOTING:
# Model A ──→ prediction A ──┐
# Model B ──→ prediction B ──┼──→ MAJORITY VOTE → final
# Model C ──→ prediction C ──┘
# (all trained independently, equal say)
#
# BAGGING:
# Data sample 1 → Tree 1 ──→ pred 1 ──┐
# Data sample 2 → Tree 2 ──→ pred 2 ──┼──→ AVERAGE → final
# Data sample 3 → Tree 3 ──→ pred 3 ──┘
# (same algo, different data, parallel training)
#
# BOOSTING:
# Tree 1 → finds errors → Tree 2 fixes them →
# Tree 2 → finds errors → Tree 3 fixes them →
# ... → WEIGHTED SUM → final
# (sequential, each focuses on previous mistakes)
#
# STACKING:
# Model A ──→ prediction A ──┐
# Model B ──→ prediction B ──┼──→ META MODEL → final
# Model C ──→ prediction C ──┘
# (like voting but a trained model decides the weights,
#  not a simple majority)
# ============================================================


# ============================================================
# 1. THE TWO MAIN FAMILIES — BAGGING vs BOOSTING
#
# A. BAGGING (Bootstrap Aggregating)
#
# The Concept: You train many models (like 100 Decision
# Trees) IN PARALLEL. Each tree sees a slightly different
# version of the data.
#
# The Goal: REDUCE VARIANCE (Overfitting).
# Famous Example: Random Forest.
#
# How it works: Each tree votes on the result. For
# regression (like your diabetes task), we take the AVERAGE
# of all 100 trees.
#
# B. BOOSTING
#
# The Concept: You train models SEQUENTIALLY (one after
# the other). The second tree tries to fix the mistakes
# made by the first tree.
#
# The Goal: REDUCE BIAS (Underfitting).
# Famous Examples: AdaBoost, Gradient Boosting, XGBoost.
#
# How it works: It puts MORE WEIGHT on the "difficult"
# data points that the previous models got wrong.
#
# VISUAL — parallel vs sequential:
#
# BAGGING (parallel — all trees train simultaneously):
# ┌─────────────────────────────────────┐
# │ Tree1  Tree2  Tree3  ...  Tree100   │ → average
# │  ↑      ↑      ↑           ↑        │
# │ sample1 sample2 sample3  sample100  │
# └─────────────────────────────────────┘
#
# BOOSTING (sequential — each tree waits for previous):
# Tree1 → mistakes → Tree2 → mistakes → Tree3 → ... → sum
#   ↑                  ↑                  ↑
# (easy points)   (focuses on      (focuses on
#                  Tree1's errors)   Tree2's errors)
# ============================================================


# ============================================================
# 2. WHY MOVE FROM A SINGLE TREE TO AN ENSEMBLE?
#
# You might have noticed that your single Decision Tree was
# very sensitive. If you changed ONE ROW of data, the whole
# tree might look different. This is called HIGH VARIANCE.
#
# Ensembles fix this by:
#
# STABILITY: One "crazy" tree won't ruin the prediction
# because 99 other trees will balance it out.
#
# HIGHER ACCURACY: They almost always outperform a single
# tree on real-world datasets like Diabetes.
# ============================================================


# ============================================================
# 4. WHAT YOU WILL NOTICE
#
# When you run this, compare the Score values to your
# single Decision Tree.
#
# In a single tree, the top feature might have a score of
# 0.50 — one feature DOMINATES.
#
# In a Random Forest, the scores are often "smoother"
# (e.g. 0.30 and 0.25). This is because the forest is
# exploring more features across different trees, giving
# you a more RELIABLE view of which features actually
# matter.
# ============================================================


# ============================================================
# WHY ENSEMBLE LEARNING WORKS
#
# The reason ensemble learning works is that it combines
# the STRENGTHS of multiple models while MITIGATING their
# weaknesses. By averaging or voting across many models,
# ensembles can reduce the impact of any single model's
# errors, leading to more accurate and robust predictions.
#
# This is particularly effective when the individual models
# are DIVERSE and make DIFFERENT TYPES of errors — the
# ensemble can "smooth out" these errors to arrive at a
# better overall prediction.
#
# ITS DISADVANTAGE IS THAT IT CAN BE MORE COMPUTATIONALLY
# EXPENSIVE THAN USING A SINGLE MODEL, AND IT CAN ALSO BE
# MORE DIFFICULT TO INTERPRET THE RESULTS, AS YOU ARE
# COMBINING THE PREDICTIONS OF MANY MODELS RATHER THAN
# RELYING ON THE INSIGHTS FROM A SINGLE MODEL.
# ============================================================


# ============================================================
# BENEFITS — BIAS AND VARIANCE REDUCTION
#
# BENEFITS ==> IT REDUCES BOTH BIAS AND VARIANCE
#
# BIAS: This is when your model is too simple and can't
# capture the underlying patterns in the data (underfitting).
# BOOSTING helps reduce bias by focusing on the mistakes
# of previous models.
#
# VARIANCE: This is when your model is too complex and
# captures noise in the data (overfitting).
# BAGGING helps reduce variance by averaging the predictions
# of many models, which smooths out the noise.
#
# SUMMARY:
# ┌─────────────┬────────────────┬──────────────────────────┐
# │ Method      │ Fixes          │ How                      │
# ├─────────────┼────────────────┼──────────────────────────┤
# │ Bagging     │ High Variance  │ Parallel training on     │
# │             │ (Overfitting)  │ different data samples,  │
# │             │                │ average predictions      │
# │ Boosting    │ High Bias      │ Sequential training,     │
# │             │ (Underfitting) │ each model fixes previous│
# │             │                │ model's mistakes         │
# │ Voting      │ Both slightly  │ Majority vote across     │
# │             │                │ independent models       │
# │ Stacking    │ Both slightly  │ Meta-model learns which  │
# │             │                │ base model to trust more │
# └─────────────┴────────────────┴──────────────────────────┘
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Ensemble = combining multiple models for better
#    prediction than any single model alone
#
# 2. Four types:
#    Voting   → majority vote, independent models
#    Bagging  → same algo, different data, parallel
#    Boosting → same algo, sequential, fixes mistakes
#    Stacking → meta-model assigns weights to base models
#
# 3. Bagging reduces VARIANCE (overfitting)
#    Boosting reduces BIAS (underfitting)
#
# 4. Diversity is the KEY — ensembles work best when
#    individual models make DIFFERENT errors
#
# 5. DISADVANTAGES:
#    → More computationally expensive
#    → Harder to interpret (no single tree to visualize)
#
# 6. INTERVIEW TRAP:
#    Q: "What is the difference between Bagging and
#        Boosting?"
#    Right answer: "Bagging trains models IN PARALLEL on
#    different random samples of data and averages their
#    predictions — it reduces VARIANCE and prevents
#    overfitting. Boosting trains models SEQUENTIALLY where
#    each new model focuses on the mistakes of the previous
#    one — it reduces BIAS and fixes underfitting. Random
#    Forest is Bagging, XGBoost is Boosting."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Random Forest
#    The most famous Bagging algorithm — combining hundreds
#    of Decision Trees trained on random data samples and
#    random feature subsets.
# ============================================================