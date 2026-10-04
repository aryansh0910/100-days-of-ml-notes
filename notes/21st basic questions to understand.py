''' we generally ask some basic quesitons from our dataset to understand it in the beginning'''
'''1. HOW BIG IS THE DATASET'''
df.shape()#this will tell u how big ur dataset is 
'''2. HOW DOES THE DATA LOOKLIKE'''
df.head()#will display the top 5 rows
df.sample(5)#it will give you any 5 random rows from the dataset it is usegul in case of bias like the bottom and top of the dataset is different
# so u use sample to check if every row looks similar 
''' to chck the datatype of each columns'''
df.info()#it will give u datatype as well as the non null values
''' checking the missing values '''
''' U CAN FILL THE MISSING VALUES USING THE SIMPLEINPUTER FUCNTION OF THE SCIKIT LERAN'''
df.isnull().sum()#this will summ up all the values  
''' how does the data look mathematicall'''
df.descirbe()#this will give u a mathematical summary of each numerical column
''' are there any fuplicate values'''
df.duplicated().sum()
''' to find correlation b/w the columns'''
df.corr()#this willgive you the correlation of each column with all the columns
