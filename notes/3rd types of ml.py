''' 
furhter ml is divided into 4 things based on learning i.e supervised learnign, unsupervised learning, semi supervised learning, reinforcement learnig
1. Supervised Learning (The "Teacher-Student" Model
it finds the mathematical relationship b/w the input and the output data so that when we provide it with new data it would tell us about
that output using that pattern
You provide the model with both the Input (Features) and the Output (Labels). It's like a student practicing with an answer key at the back of the book.

Regression: Predicting a continuous number (e.g., Car Price, Stock Market trends).
Classification: Predicting a discrete category (e.g., Is this email "Spam" or "Ham"? Is this tumor "Malignant" or "Benign"?).


2. Unsupervised Learning (The "Pattern Finder")
it has no output data the model itsellf has to find the patterns
Here, you give the model Input data but no labels. The model has to find hidden structures or groups on its own.

Clustering: Grouping similar data points together (e.g., Segmenting customers into "Frequent Buyers" vs. "One-time Shoppers").
Dimensionality Reduction: Simplifying complex data by keeping only the most important features (e.g., PCA).
like in case there are many input features in case for supervised learnign then you can convert ur many features into a 2 or 3 features
without loosing much data and time by training on all those features
Association: Finding rules that link items (e.g., "People who buy milk also tend to buy bread").
Analomy detection: like if already formed a cluster and then give a point and the point lies far from any clusters then it will raise
flag like this is an exception

3. Semi-Supervised Learning (The "Hybrid")
like in google photos u label ur image with father then all the images with ur father in it wil be lablelled automatically
wihtout u doing anything

Labeling data is expensive and time-consuming (humans have to do it). Semi-supervised learning uses a small amount of labeled data 
and a huge amount of unlabeled data
Example: Google Photos. You label a few photos of your friend "John" (Supervised), and the AI then finds thousands of other 
photos of that same face across your library (Unsupervised).

4. Reinforcement Learning (The "Reward-Based" Model)
This is entirely different. There is an Agent (the AI) in an Environment. It takes Actions and receives Rewards (positive) or 
Punishments (negative). It learns to maximize its total reward over time.
Example: AlphaGo, self-driving cars learning to stay in lanes, or a robot learning to walk


Feature             Supervised                        Unsupervised                            Reinforcement
Data                Labeled (Inputs + Answers)    Unlabeled (Inputs only)           No predefined data (Interactive)
Goal                Predict/Classify              Find hidden patterns              Maximize rewards
Feedback            Direct (True labels)          No specific feedback              Delayed (Rewards/Penalties)
Common Use          "Price prediction             Spam filter","Customer segments   Anomaly detection","Games, Robotics, Navigation             
'''