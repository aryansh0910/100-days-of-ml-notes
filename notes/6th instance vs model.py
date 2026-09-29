'''
the next thing on which the machine learning is divided is that how it learns from the data 

1. Instance-Based Learning (The "Comparison" Method)THIS ONE IS LIKE MEMORIZING THE DATA
IT JUST FINDS THE SIMILARITY OF THE DATA WITH THE VALUE TO BE PREDITCED AND ANSWER ACCORDINGLY
THIS ALWAYS REQUIRE THE TRAINING DATA TO PREDICT SO IT IS MEMORY HEAVY AS IT ALWAYS NEED THE DATA
THIS IS ALSO KNOWN AS LAZY LEARNER
In Instance-based learning, the system learns the training data by heart. It doesn't try to find a general rule; it just stores
every example it has ever seen 

How it works: When you give it a new car to predict, it looks through its entire memory to find the cars that are most similar (its "neighbors").
The Logic: "This new car is very similar to these 3 cars I saw in my training data. Their average price was $5,000, so I'll guess $5,000."
Most Famous Example: k-Nearest Neighbors (k-NN). IN THIS WEHRE U SET THE N_NEIGHBOURS=3 OR 5 MEANS U WILL CALCULATE THE 3 OR 5 NEAREST
POINTS TO THE ASKED POINT I.E THE INPUT AND THEN CHECK IF ITS A YES OR NO AND IT WILL ANSWER ACCORDING TO THE MAJORITY I.E IF 2 OUT OF
3 POINTS ARE THE YES SO THE ANSEWR WILL BE YES SO THE K_NEIGHBOURS DECIDE UPTO HOW MANY CLOSEST POINTS U WANT TOO LOOK T


2. Model-Based Learning (The "Generalization" Method)
THIS TYPE OF LEARNING TRIES TO FIND THE PATTERN IN THE DATA AND ASNWER ACCORDINGLY NOT BY SEACRHING THE SIMILAR THINGS
THEY FIND A MATHEMATICAL RELATIONSHIP
SO EVEN AFTER THE DATA SET IS GONE UR MACHINE STILL REMEMBERS THAT MATHEMCATICAL RELATIONSHIP SO IT WILL ANSWER ACCORDING TO THAT ONLY

This is what you did with Linear Regression, XGBoost, and CatBoost. Instead of memorizing every car, the model tries to find a pattern or
a mathematical formula that fits the data.   

How it works: It looks at all the car data and says, "Okay, I've noticed that for every year a car ages, the price drops by 10%." 
It summarizes all that data into a few numbers (weights/parameters).

The Logic: Once the model is trained, you can throw away the original data. You only need the formula 
(e.g., Price = (W_1 * Age) + (W_2 * KM))

'''