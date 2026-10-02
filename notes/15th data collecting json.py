'''
Most company data isn't sitting in neat files; it's locked inside Databases (SQL) or flying across the web in JSON

JSON IN MOST COMMONLY USED AS IT CAN BE UNDERSTAND BY MOST LANGUAGES LIKE C++,PYTHON AND JAVA
1. Working with JSON (The Web's Language)
JSON (JavaScript Object Notation) is the standard for APIs. Unlike CSVs, which are flat (rows and columns), JSON is hierarchical (nested).

Why use JSON?
It handles the "List in a column" problem you just faced naturally.

It's the format used by Google Maps, Twitter, and Weather APIs.

The Challenge: Flattening
Real JSON is often nested (dictionaries inside lists inside dictionaries). If you just use read_json, your columns will look like "junk."

The Solution: pd.json_normalize(). This "flattens" the nested data into a standard table format.
'''
''' WHENEVER U  REQUEST DATA FROM THE API IT RETURNS DATA IN THE JSON FORMAT'''
# FOR LOADING
import pandas as pd

# Method 1: Direct Load
df = pd.read_json('data.json')

# Method 2: From a URL (API)
df = pd.read_json('https://api.exchangerate.host/latest')
