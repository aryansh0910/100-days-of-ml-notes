''' SOME IMPORTANT PARAMTERS OF CATBOOST ARE ==>
Parameter               Alias               Purpose
iterations              n_estimators        The total number of trees to build. The default is 1000.
learning_rateeta       The step size        If you have many trees (iterations), use a smaller learning_rate (e.g., 0.01) to avoid overshooting the answer."
depth                  max_depth            The depth of the Symmetric Trees. Usually between 4 and 10. Deeper trees capture complex patterns but overfit easily.


cat_features: A list of indices or names for your categorical columns. This triggers the Ordered Target Encoding we discussed.'''