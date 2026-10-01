''' 

THE FIRST AND THE FOREMOST STEP IS FRAMING THE PROBLEM IN A MACHINE LEARNING MODEL 
1. The 3 Steps of Problem Framing
Step A: Identify the "Decision"
ML should not just give a "prediction," it should help take an action.

Vague Wish: "We want to know more about our car sales." (No clear action).
Framed Problem: "We want to predict the fair market price of a car so we can suggest a listing price to the seller." (Clear action: setting a price).

Step B: Choose your "ML Lens"Once you know the goal, you decide which type of ML handles it.
Is it a number? ==> Regression (e.g., Price of a car).
Is it a category? ==> Classification (e.g., Is this car "Luxury" or "Budget"?).
Is it a group? ==> Clustering (e.g., "Which customers like SUVs?").

Step C: Define the "Success Metric"How will you prove to your boss that the model is working? You need two types of metrics:Technical
Metric: $R^2$ score or Mean Absolute Error (MAE).Business Metric: "Does using this model increase the number of cars sold per month?"

'''