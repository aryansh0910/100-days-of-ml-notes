'''
FIRST WE NEED ANOTHER LIBRARY THAT WOULD HELP PYTHON TO CONNECT TO THE SQL DATA AS IT WONT UNDERSTAND THAT
U ALSO NEED A SOFTWARE KNOWN AS XAMP TO RUN THE SQL FILE AND CONVERT IT INTO THE PANDAS DATAFRAME
1. The "Bridge" Strategy
Python cannot talk to a database directly. It needs a Database Driver (a connector).

For SQLite: Use sqlite3 (built into Python).

For MySQL: Use mysql-connector-python.

For PostgreSQL: Use psycopg2
'''
import mysql.connector
import pandas as pd

conn = mysql.connector.connect(
    host='localhost',       # Or an IP address
    user='root',            # Your DB username
    password='password123', # Your DB password
    database='car_dekho'    # The specific database name
)
query = "SELECT brand, year, selling_price FROM cars WHERE km_driven < 50000"
df = pd.read_sql_query(query, conn)
conn.close() # Always close the door when you're done!