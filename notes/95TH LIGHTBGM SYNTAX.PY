import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
X=input
y="output"

# 1. Split your data as usual
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Initialize the model 
# (Start with these basic parameters)
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,

)
''' now to use the feature for the categorical columns you have to first onvert the data type of that column to the category'''
df["fuel_type"]=df["fuel_type"].astype("category")
''' in case of multiple columns u can use for loop'''
# Step 1: Create a list of the columns you want to change
cat_cols = ['fuel', 'transmission', 'owner', 'brand']

# Step 2: The loop starts here
for col in cat_cols:
    df[col] = df[col].astype('category')
# 3. Fit the model
model.fit(X_train, y_train,categorical_feature=["columns name you want to use inplace of label encoder"])#u can use this inplace of pd.get dummies


# 4. Predict and Evaluate
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")