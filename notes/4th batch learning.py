'''
batch learning is also known as offline learning.
MEANS U WILL TRAIN THE DATA ON UR MACHINE AND THEN DEPLOY IT ON THE SERVER RATHER THAN TRAINING ON THE SERVER AS THAT WOILD BE TIME TAKING
AND ALSO COSTLY
Batch Learning is a system where the model is incapable of learning incrementally. It must be trained on all available data at once.
Whenever we train our model we train it on a specific data that is given so if in future something new is added or something
is changed like the price of the stocks our model would still be stcuk with the old data giving the old results


1. How the Workflow Looks

Imagine you are building your Car Dekho model. You have a CSV file with 5,000 rows.

You take all 5,000 rows.
You train the model (the "Batch").
You launch the model into production.

The model is now "frozen." If 1,000 new cars are sold tomorrow, the model doesn't know about them. It will keep predicting based
on the old 5,000 rows until you manually stop everything and retrain it from scratch using all 6,000 rows.

Data Drifting: If the market changes (e.g., car prices suddenly drop due to a recession), a batch model will start giving wrong 
answers because it’s "stuck in the past." This is called Model Decay.

NOW WHAT CAN YOU DO IS SCHEDULING A NEW PROGRAM FOR COLLECTING DATA AND COMBING IT AND TRAINING AGAIN 

. The "Senior Dev" Perspective
In most internships, you will start with Batch Learning. However, to prevent your model from becoming "dumb" over time, 
companies set up Automated Pipelines.

For example, a script might run every Sunday at 2 AM that:
Pulls the new data from the database.
Combines it with the old data.
Retrains the model.
Replaces the old model with the new one.

'''