'''
 FETCHING THE DATA FROM THE API IS how you get real-time, dynamic data like live stock prices, weather updates, or
 movie metadata directly into your Python environment.
 API ==> APLICATION PROGRAMMING INTERFACE API SIMPLY HELPS TWO SOFTWARE TO WORK TOGETHER OR U CAN SAY COMMUNICATE WITH EACH OTHER
 THEY ARE ALSO KNOWN AS DATAPIPLINES WHICH TRANSFERS THE DATA

Think of an API as a waiter in a restaurant. You (the client) tell the waiter what you want, the waiter goes to the kitchen 
(the server/database), and brings the food (the data) back to your table.

1. The Core Toolkit: requests library
In Python, we use the requests library to "call" an API. It's not a built-in library
Most APIs use the GET method to send data. Here is the standard workflow:

Step 1: The Request
You send a "ping" to the API URL.
'''
import requests
import pandas as pd

url = "https://api.themoviedb.org/3/movie/top_rated?api_key=YOUR_KEY"
response = requests.get(url)
'''Step 2: Check the Status Code
Before processing, always check if the "ping" was successful.

200: Everything is fine (OK).

401: Unauthorized (Your API key is wrong).

404: Not Found (The URL is wrong).

Step 3: Parse the JSON
APIs usually return data in JSON format. We convert it into a Python dictionary.
data = response.json()

Step 4: Extract the Relevant Part
API responses often contain "meta-data" (like page numbers or total results). You need to find the specific list of data you want

# Usually, the actual data is under a key like 'results' or 'data'7
movies_list = data['results']

Step 5: Load into Pandas
df = pd.DataFrame(movies_list)'''