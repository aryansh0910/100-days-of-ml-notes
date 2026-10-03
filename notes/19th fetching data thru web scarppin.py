''' 
to fetch the data from the web we have to use a library called bs4 and use its beautiful soup functioon for it
Beautiful Soup is the "translator" that takes messy, confusing HTML code and turns it into a structured, searchable Python Object
1. The Core Concept: The "Parse Tree"
When you create a soup object, BeautifulSoup analyzes the HTML and builds a Tree. In this tree, every tag (like <div>, <h1>, <a>) is a
 "branch" or "leaf" that you can reach using Python commands
 
 
 from bs4 import BeautifulSoup

# 'lxml' is the engine that does the heavy lifting.
# It is faster than the default 'html.parser'.
soup = BeautifulSoup(webpage, 'lxml')

2. The Four Main Objects
Every piece of data you extract with BeautifulSoup will be one of these four types:

Object Type                 Description                      Example
BeautifulSoup               The entire document.                 soup
Tag                         A specific HTML element.             soup.h2 or soup.div
NavigableString             The actual text inside a tag.         soup.h2.string
Comment                     A special string for HTML comments     ``'''
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
''' now we will try to fetch the data from that website using the requests library '''
response=requests.get("https://www.ambitionbox.com/list-of-companies?campaign=desktop_nav")
print(response)# this will give us respone 403 neans the access is denied bcoz most websites dont want their site to be scrapped so
# we have to look like a human to that website not like a bot trying to scrap its website for that we will use so tools known as header
print(response.text) #this will display the html code of the respoonse
'''the requests library sends a default identification string that basically yells, "I am a Python script!" (usually something 
like python-requests/2.31.0). Most modern websites see this and immediately block your connection to protect their servers from
bots'''
'''Why You Need Headers
To Look Like a Human: By adding a User-Agent, you "cloak" your script so it looks like a real person using Chrome, Firefox, or Safari.

To Avoid 403 Forbidden Errors: Many sites will return a 403 error the moment they see a request without a proper browser header.

To Get the Right Content: Some sites serve different versions of their page to mobile vs. desktop users. Headers tell the server which version you need.'''

''' now we wiill use header so we can get acess'''
'''# This tells the server you are using Chrome on a Windows 10 PC
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
}'''#this header is enough
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
}
response=requests.get("https://www.ambitionbox.com/list-of-companies?campaign=desktop_nav",headers=headers)
print(response)# now the response will be 200

''' now u will convert ur response into the html code using the text function and store it in a webpage'''
webpage=response.text
# print(webpage)#this file would be containg some emojis or special characters which are not encoded  so u can rewrite the file with utf8 encoding and save it in html format
with open("19_webpage.html","w+",encoding="utf-8") as f:
    f.write(response.text)#this will create a new file with the name webpage.html and we will write the response.text o.e the html code in it
# ''' this is the html code of the webpage i.e is know saved inside our computer'''
# ''' now we will use that webpage and store it in a variablle known as webpage to perform the further tasks'''
with open("19_webpage.html","r",encoding="utf-8") as f:
    webpage=f.read()
    print(webpage[1:100])#this type of data will be in the webpage 
    ''' now we will use the beautiful soup to extract the data'''
    soup=BeautifulSoup(webpage,"lxml")#now using the library to extract everyhitng# 'lxml' is the engine that does the heavy lifting.
# It is faster than the default 'html.parser'.
    soup.prettify#this will make the code in the a struuctured way 

    ''' now for further u need the html knowledge'''

