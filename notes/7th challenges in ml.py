'''
we usually face many challenges in ml some of them are
1. Insufficient Quantity of Training Data
Machine Learning is "data-hungry." Even for simple tasks, you need thousands of rows. For complex tasks like Image Recognition, you need millions.
The Reality: If you only have 50 car records, your model will never learn the pattern; it will just guess.
The Solution: Data Augmentation (creating fake data that looks real) or simply collecting more.

2. Non-Representative Training Data
Your model is only as smart as the data it sees. If your training data doesn't represent the "real world," your model will be biased.
The Scenario: You train your car model using only Luxury Cars (BMW, Audi). Then, someone uses your app to predict the price of a Maruti 800.
The Result: The model will give a ridiculously high price because it thinks all cars are expensive. This is called Sampling Bias.

3. Poor-Quality Data (The "Garbage In, Garbage Out" Rule)
MEANS IF U ADD UNNECESSAY FEATURES IN UR MODEL THEN THE RESULTS WILL ALSO BE UNNECESSARY
If your data is full of errors, outliers, and noise, the model will get confused.

Common Issues:
Outliers: A car listed for $1 when it’s worth $10,000 (typo).
Missing Values: 30% of cars don't have the "Kilometers Driven" listed.
The Solution: Data Cleaning (imputing missing values, removing duplicates, or filtering out-of-range outliers).

4. Irrelevant Features (Feature Engineering Challenge)
Not all columns are useful. Including "junk" columns makes the model slower and less accurate.
The Example: Including the "Color of the Car's Seat Covers" or "Owner's Name" to predict the selling price. These don't help and just add "noise."
The Pro Move: Feature Selection—the art of picking only the columns that actually impact the target.

5. Overfitting vs. Underfitting (The "Final Boss")
This is the most important part of this day's lesson.
OVERFITTING MEANS THE MODEL ONLY PERFORMS GOOD ON THE GIVEN DATA NOT THE REAL WORLD DATA MEANS IT MEMORIZE THE ORIGINAL GIVEN DATA
RATHER THAN FINDING PATTERNS

Overfitting (The "Ratta" Student): The model performs perfectly on training data but fails on new data.
It has "memorized" the noise rather than the pattern.
Analogy: A student who memorizes every question in the textbook but fails if the teacher changes even one number in the exam.
Underfitting (The "Lazy" Student): The model is too simple to even learn the training data.
Analogy: Trying to use a straight line to predict a relationship that is clearly a curve

'''
