''' 
first you have to check a website with a api and find its api
we will first use the tmdb.api website and find the api of the top rated movies and the fill in our api key
the api key will look like this 8265bd1679663a7ea12ac168da84d2e8
the the api link will look like this after filling the api keey ==>
https://api.themoviedb.org/3/movie/top_rated?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US&page=1


 now the data will look liike the json format data on the website THE THING THAT YOU WILL BE Seeing IS NOT THE WEB PAGE ITS THE DATA wheen u click on the api url
 YOU CANT DIRECTLY UNDERSTAND THE JSON DATA SO U HAVE TO PASTE ALLL THE DATA IN A WEBSITE KNKOWN AS JJSON VIEWER WHICH WILL TELL YOU ABOUT THE
 JSON FILE
 IT WILL TELL YOU ABOUT THE TOTAL PAGES THE TOTAL RESULTS
 JSON JUST LOOKS LIKE THE PYTHON DICTIONARY
 NOW U WILL CREATE A NEW DATAFRAME FROM THIS JSON DATA
 YOU CAN SELECT THE COLUMNS U WANT OR CAN DIRECTLY GET ALL THE COLUMNS AS WELL
 THE DATASET MAY CONSIST OF MANY PAGES LIKE IN THE END OF THE URL THE PAGE NO. IS WRITTEN SO EVERYTIME WE HAVE TO CHANGE THAT URL 
 AND IT WILL WORK UPTO THE TOTAL NO. PAGES USING THE FOR LOOOP AND U WILL KEEP APPENDING THAT DATAFRAME USING THE FOR LOOP AND IN THE
 END YOU WILL BE ABLE TO SEE ALL THE DATA 
'''
import pandas as pd
import numpy as np
import requests#this will allow you to use the api link i.e the url
#now u will use the get url function of the requests liibrary and paste the link inside it
# requests.get("https://api.themoviedb.org/3/movie/top_rated?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US&page=1")
# as tmdb was not responding so we used omdb
response = requests.get("https://www.omdbapi.com/?i=tt3896198&apikey=6ac8fb6f")# now an request will be sent on the api given o.e this link
# and then it will be able to fetch the data i.e is written on the api
print("PRITINH THE ACTUaL RESPONSE")
print(response)#response [200] means everything is okay if [404] then there would be error it means
# now we have to convert that response into the json data using response.json
print("PRINting the RESpIOnse oF JSON")
print(response.json())# this willl display everything in the json format like given on the api
response=response.json()#now this json file is like a dictionary it would have key : value pairs
df=pd.DataFrame(response)# now we will convert it into the dataframe
print("PRINTING THE DATAFRAME")
print(df)
''' THE DIFFERENCE B/W THE DATAFRAME AND THE NORMALIZE IS THAT THE DATAFRAME DOESNT CARE ABOUUT THE NESTED LISTS OR DICTINIRES
IF THEY ARE THE VALUE OF ANY KEY SO IT JUST FITS EVERYTHING IN A SINGLE BOX LIKE IN THIS CASE WITH THE RATINGS COLUMN
BUT THE JSON.NORMALIZE CREATES A DIFFERENT CLUMN FOR EACH OF THIS'''
print("PRINTING THE DATAFRAME RATINGS")
print(df["Ratings"].head())
dd=pd.json_normalize(response,record_path=["Ratings"],meta=["Title","Year"],errors="ignore")#this recored_path will tell the python which is a nested list
#the meta will tell the json_normalize to attach the other columns as welll i.e are already there the errors ignore will feel the nan value if something is not given
print("PRINTING THE JSON NORMALIZE COLUMN")
print(dd.columns)
print(dd.head())
print(dd["Year"].value_counts())

