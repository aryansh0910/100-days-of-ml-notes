# ============================================================
# OPTUNA — BAYESIAN HYPERPARAMETER OPTIMIZATION
# ============================================================

# ============================================================
# WHAT IS IT?
#
# Optuna is the "modern" way to tune models. Unlike
# GridSearchCV (which tries every single combination like
# a brute) or RandomSearchCV (which just guesses), Optuna
# uses BAYESIAN OPTIMIZATION. It "thinks" about previous
# results and narrows in on the best parameters.
#
# ITS SIMILAR TO GRIDSEARCHCV AND RANDOMSEARCHCV BUT WAY
# MORE EFFICIENT AND SMARTER. IT USES A TECHNIQUE CALLED
# "BAYESIAN OPTIMIZATION" TO LEARN FROM PREVIOUS TRIALS
# AND FOCUS ON THE MOST PROMISING AREAS OF THE
# HYPERPARAMETER SPACE.
#
# ITS LIKE A SMART SHOPPER WHO TRIES A FEW SAMPLES OF
# DIFFERENT FLAVORS OF ICE CREAM, THEN DECIDES TO FOCUS
# ON THE FLAVORS THEY LIKED THE MOST INSTEAD OF TASTING
# EVERY SINGLE FLAVOR IN THE STORE.
#
# OPTUNA ESTABLISHES A MATHEMATICAL MODEL OF THE
# HYPERPARAMETER SPACE BASED ON THE RESULTS OF PREVIOUS
# TRIALS AND USES THIS MODEL TO SELECT THE MOST PROMISING
# HYPERPARAMETERS FOR THE NEXT TRIAL. THIS LEADS TO FASTER
# CONVERGENCE TOWARDS THE BEST PARAMETERS COMPARED TO
# GRIDSEARCHCV OR RANDOMSEARCHCV.
# ============================================================


# ============================================================
# COMPARISON — GridSearchCV vs RandomSearch vs Optuna
#
# ┌──────────────────┬──────────────────┬───────────────────┐
# │                  │ GridSearchCV     │ Optuna            │
# ├──────────────────┼──────────────────┼───────────────────┤
# │ Strategy         │ Try EVERY combo  │ Learn from past   │
# │                  │ (brute force)    │ trials, focus on  │
# │                  │                  │ best areas        │
# │ Efficiency       │ Low — 6×6×6=216  │ High — 30 smart   │
# │                  │ trials minimum   │ trials often beat │
# │                  │                  │ 216 blind ones    │
# │ Pruning          │ No — all trials  │ Yes — stops bad   │
# │                  │ run to the end   │ trials early      │
# │ Visualization    │ None built-in    │ Built-in plots    │
# └──────────────────┴──────────────────┴───────────────────┘
# ============================================================


# ============================================================
# IMPORTANT TERMS IN OPTUNA
#
# STUDY: The overall optimization task.
#
# TRIAL: A single execution of the model with ONE set of
# parameters.
#
# TRIAL PARAMETERS: The specific hyperparameters that were
# tested in that trial.
#
# OBJECTIVE FUNCTION: A small function you write that tells
# Optuna what score to MAXIMIZE (like Accuracy) or
# MINIMIZE (like Error).
#
# SAMPLING METHODS: Optuna uses different algorithms to
# decide which hyperparameters to try next (after checking
# previous):
# → TPE (Tree-structured Parzen Estimator) — DEFAULT
# → CMA-ES (Covariance Matrix Adaptation) — for high dims
# → Random Sampler — just like RandomSearchCV
# ============================================================


# ============================================================
# SUGGEST METHODS — HOW TO DEFINE PARAMETER RANGES
#
# ┌────────────────────────────┬───────────┬──────────────────┐
# │ Method                     │ Type      │ Use Case         │
# ├────────────────────────────┼───────────┼──────────────────┤
# │ trial.suggest_int          │ Integer   │ max_depth,       │
# │                            │           │ n_estimators     │
# │ trial.suggest_float        │ Float     │ learning_rate,   │
# │                            │           │ alpha            │
# │ trial.suggest_categorical  │ List      │ kernel='rbf' or  │
# │                            │           │ kernel='linear'  │
# │ trial.suggest_float(log=T) │ Log Scale │ C, gamma (vary   │
# │                            │           │ by magnitudes)   │
# └────────────────────────────┴───────────┴──────────────────┘
# ============================================================


# ============================================================
# 3. WHY OPTUNA IS BETTER THAN GRIDSEARCHCV
#
# PRUNING: Optuna can STOP a "bad" trial halfway through
# if it looks like it's going to fail, saving hours.
#
# EFFICIENCY: Finds the "Goldilocks" zone much faster
# because it LEARNS FROM ITS MISTAKES.
#
# VISUALIZATION: Built-in plots show which parameters
# actually matter.
#
# VISUAL — how Bayesian optimization works:
#
# Trial 1: n_estimators=100, lr=0.3 → score=0.72
# Trial 2: n_estimators=500, lr=0.1 → score=0.85
# Trial 3: Optuna thinks "higher n_estimators + lower lr
#           seems promising" → tries n_estimators=700, lr=0.05
#           → score=0.89 ✅
#
# GridSearchCV would have tried ALL combinations blindly.
# Optuna focused on the promising area after 2 trials!
# ============================================================


# ============================================================
# CODE — SINGLE MODEL TUNING (XGBoost)
# ============================================================

import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# 1. Define the objective (The 'Goal')
def objective(trial):

    # 2. Define model and suggest values directly in args
    # SUGGEST_INT means it will try different integer values
    # between the range AFTER CHECKING PREVIOUS TRIALS
    model = XGBClassifier(
        n_estimators  = trial.suggest_int('n_estimators',
                                           100, 1000),
        max_depth     = trial.suggest_int('max_depth', 3, 9),
        learning_rate = trial.suggest_float('learning_rate',
                                             0.01, 0.3),
        subsample     = trial.suggest_float('subsample',
                                             0.5, 1.0),
        random_state  = 42
        # trial.suggest_categorical('booster',['gbtree','dart'])
        # trial.suggest_float('gamma', 1e-5, 1e-1, log=True)
        # log=True → searches on log scale for params that
        # vary by orders of magnitude (like C or gamma)
    )

    # 3. Get the score using Cross-Validation
    # Splits data into 3 parts and averages the accuracy
    score = cross_val_score(model, X_train, y_train,
                             cv=3).mean()
    return score

# 4. Create the study and start the search
# direction='maximize' → Optuna tries to get highest score
# TPESampler is DEFAULT and most commonly used
# CMA-ES is better for high-dimensional problems
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler()
)

study.optimize(objective, n_trials=20)
# RUN 20 TRIALS TO FIND THE BEST HYPERPARAMETERS
# (increase to 100 for larger datasets)

# 5. Print the winners
print("Best Accuracy found:", study.best_value)
print("Best Settings:", study.best_params)


# ============================================================
# CODE — MULTI-MODEL TUNING (ULTIMATE WAY)
#
# U CAN ALSO USE OPTUNA TO FIND THE BEST MODEL TYPE BY
# USING TRIAL.SUGGEST_CATEGORICAL TO CHOOSE BETWEEN
# DIFFERENT ALGORITHMS (LIKE SVM, RANDOM FOREST, XGBOOST)
# AND THEN TUNE THEIR HYPERPARAMETERS SIMULTANEOUSLY.
# ============================================================

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 1. Define the Master Objective
def objective(trial):
    # Optuna first picks WHICH model to try
    classifier_name = trial.suggest_categorical(
        "classifier", ["SVM", "RandomForest", "XGBoost"]
    )

    # 2. Set up the specific 'knobs' for each model
    if classifier_name == "SVM":
        svc_c = trial.suggest_float("svc_c", 0.1, 10.0,
                                     log=True)
        model = SVC(C=svc_c, kernel='rbf')

    elif classifier_name == "RandomForest":
        rf_depth = trial.suggest_int("rf_max_depth",
                                      2, 32, log=True)
        model = RandomForestClassifier(max_depth=rf_depth,
                                        n_estimators=100)

    else:  # XGBoost
        xgb_depth = trial.suggest_int("xgb_max_depth", 3, 9)
        xgb_lr    = trial.suggest_float("xgb_learning_rate",
                                         0.01, 0.3)
        model = XGBClassifier(max_depth=xgb_depth,
                               learning_rate=xgb_lr,
                               n_estimators=100)

    # 3. Score using 3-Fold Cross Validation
    score = cross_val_score(model, X_train, y_train,
                             n_jobs=-1, cv=3).mean()
    return score

# 4. Run the Search
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# 5. Output the Results
print("Winner Model and Params:")
print(study.best_params)
print("Best Cross-Validation Score:", study.best_value)

# Build the final model with the best params
# e.g. if best_params says classifier='XGBoost',
#         xgb_max_depth=5, xgb_learning_rate=0.1
final_model = XGBClassifier(max_depth=5,
                              learning_rate=0.1,
                              n_estimators=100)
final_model.fit(X_train, y_train)
print("Final Test Accuracy:", final_model.score(X_test,
                                                  y_test))


# ============================================================
# WHY THIS IS THE "ULTIMATE" WAY TO WORK
#
# ZERO GUESSING: You aren't just guessing if XGBoost is
# better than SVM — the math PROVES it.
#
# EFFICIENCY: It doesn't try random settings; it tries
# settings that WORKED WELL in previous trials.
#
# FAIR COMPARISON: By using cross_val_score inside the
# loop, no model wins just because it got a "lucky" split.
#
# PRO TIP: If you have a lot of data, change n_trials=30
# to n_trials=100 to let Optuna explore more combinations.
# ============================================================


# ============================================================
# KEY POINTS TO REMEMBER
#
# 1. Optuna = smart hyperparameter search that LEARNS from
#    previous trials (Bayesian optimization) vs GridSearch
#    which blindly tries every combination
#
# 2. Key terms:
#    Study     → the overall search task
#    Trial     → one run with one set of params
#    Objective → function returning the score to optimize
#
# 3. Suggest methods:
#    suggest_int        → integers (n_estimators, max_depth)
#    suggest_float      → floats (learning_rate, subsample)
#    suggest_categorical → choices (kernel, booster)
#    suggest_float(log=True) → log scale (C, gamma)
#
# 4. TPESampler (default) → learns from past trials
#    CMA-ES → better for high-dimensional spaces
#    Random → just like RandomSearchCV (no learning)
#
# 5. Pruning → Optuna stops bad trials early, saving time
#    GridSearchCV always runs every trial to completion
#
# 6. INTERVIEW TRAP:
#    Q: "Why use Optuna over GridSearchCV?"
#    Right answer: "GridSearchCV tries every single
#    combination blindly — if you have 5 values for each
#    of 4 params, that's 625 trials. Optuna uses Bayesian
#    optimization to LEARN from each trial result and
#    focus future trials on the most promising parameter
#    regions. It also has PRUNING — if a trial looks bad
#    halfway through, it stops early rather than wasting
#    time finishing it. This means 30 smart Optuna trials
#    often beat 625 blind GridSearch trials both in speed
#    and final accuracy."
# ============================================================


# ============================================================
# COMING UP NEXT:
# → Model Deployment with Streamlit
#    Saving your tuned model and deploying it as a web app
#    for your Resume Screener project.
# ============================================================