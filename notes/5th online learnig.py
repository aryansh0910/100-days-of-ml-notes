''' 
Online Learning (also known as Incremental Learning).
IN ONLINE LEARNING LIKE THE OFFLINE LEARNING THE MODEL IS DEPLOYED ON THE SERVER AFTER TRAINING THE MODEL ON THE MACHINE BUT THE 
CHANGE IS THAT THE NEW DATA DIRECTLY COMES TO THE SERRVER AND THE MODEL THEN LEARNS AS WELL AS PREDICTS FROM THE NEW DATA IN THE MININ 
BATCHES WHILE IN OFFLINE LEARNING THE MODEL ON THE SERVER JUST PREDICTS THE DATA NOT LEARN FROM IT WE HAVE TO GO OFFLINE IN THAT IF WE
WANT OUR MODEL TO LEARN
SGDREGSSOR FROM SCIKITLEARN.LINEAR_MODEL IS AN COMMON EXAMPLE OF ONLINE LEARNING 
The Core Concept: Learning in Chunks
In Online Learning, the system is trained incrementally by feeding it data instances sequentially, either individually or in small
groups called mini-batches.
The Loop: A new data point arrives ==> The model makes a prediction ==> It learns the correct answer
==> It updates its weights ==> The data point is discarded.Why it's cool: The model "lives" in the present.
It doesn't need to wait for a massive update; it changes as the world changes.


Learning Rate ($\eta$): This is the most critical setting. It controls how fast the model "forgets" old data to learn new data.
High $\eta$: Adapts fast, but is erratic and forgets old patterns quickly.Low $\eta$: Stable, but slow to react to new trends.
Partial Fit: Instead of the .fit() method you used for your Car Dekho project, online models use a method called .partial_fit()


THE PARTIAL FIT MEANS THAT UR DATA WILL BE ADDED IN THE PREVIOUS DATA TO COMBINE THE BOTH AND TRAIN THEM.
LEARNING RATE MEANS HOW OFFENTLY U WANT UR MODEL TO UPDATE THE DATA AND LEARN FROM IT
IT IS ALSO BENEFICAL IN CASE WHERE THE ORIGINAL DATA IS MORE THAN UR SYSTEM CAN HANDLE AT THAT TYM U CAN USE OUT OF CORE DATA CONCEPT
I.E PROVIDE DATA INCREMENTALLLY

'''
 