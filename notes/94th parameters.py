'''
PARAMTERS ==>
num_leaves ==>  its same as the max_depth means how many trees should be grown
High value: More accurate, but high risk of overfitting (memorizing).
Rule of thumb: Set it to 2^{max_depth}. So if max_depth is 5, num_leaves should be around 31.


max_depth ==> Even though LightGBM grows leaf-wise, you can still put a "ceiling" on how tall the tree gets. 
This is a safety net to keep the model from getting too crazy.

learning_rate ==> This is the "step size." If you set it to 0.1, the model learns 10% of the pattern from each tree

n_estimators ==> The number of trees the model will build. If the learning rate is small, you need more trees.

categorical_feature ==> As we discussed, you can pass a list of column names (like ['brand', 'fuel']) here.
LightGBM will then use its special math to handle them without needing get_dummies.this is use to convert multiple columns like 0 and 1 for 
petrol and diesel into a single column with 0 and 1 for petrol and 2 and 3 for diesel means if the value is 3 diesel is present
if the value is 1 petrol is presnent

Goal                  Parameter to adjust         Direction
Increase Accuracy       num_leaves               Increase ↑
Stop Overfitting        min_data_in_leaf        Increase ↑
Stop Overfitting        max_depth                Decrease ↓
Train Faster            learning_rate            Increase ↑
Be More Precise         learning_rate            Decrease ↓

SETTING THE VALUES
n_estimators: 100 to 1000. Start with 100 to see if it works, then increase it.

learning_rate: 0.01 to 0.1.
Tip: If you use a low rate (0.01), you must increase n_estimators (to 1000+) so the model has enough time to learn.
num_leaves: 31. This is the default and is usually plenty for medium datasets.

max_depth: Set this to 5, 7, or 9. This forces the trees to stop growing too tall.

min_data_in_leaf: Set this to 20, 50, or 100.

If you set it to 100, the model won't create a "price rule" unless it sees at least 100 similar cars. This stops it from memorizing "one-off" weird prices.

. Tuning for High Accuracy (Big Data)
If you have a massive dataset (50,000+ rows) and the model is too simple, "unleash" it:

num_leaves: Increase to 64, 128, or even 255.

max_bin: By default, this is 255 (the "buckets" we talked about). If you increase this to 510, the model sees more detail in the
 numbers, but it will be slower.
'''
