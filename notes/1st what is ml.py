''' in ml we give our program/model a data to find patterns and perform a specific set of tasks on it and find the output if
we give them a new data 
for example ==>
In traditional programming, you are the Boss. In Machine Learning, you are the Teacher

 you write a program for sum of 2 numbers and then u want to find the sum of 10 numbers for that you will have to change ur whole program
 but in case of ml this would not happen youu give your model the data and the output it will find the pattern that ur doing addition
 then it will automatically add all the no. u add in ur input without doing any change in it
 similar is the case with the email spam or not normally thru a proggram u would have to use many if else conditions like
 if this word comes then it is a spam if this comes it is not so if the spam email company changes that word ur program will fail
 or u have to make many changes for that word but in ml model ur model will catch that and it will automatically send them in spam as well
 '''
import time

start = time.time()
# Perform a simple math loop 10 million times
x = [i**2 for i in range(10_000_000)]
end = time.time()

print(f"Time taken: {round(end - start, 2)} seconds")