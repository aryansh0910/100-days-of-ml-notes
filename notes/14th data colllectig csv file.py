''' 
the model we prepare depends on the data we give it if we give it the good data it is highly probable that the model wiill be
good as well and it will predict good too
there are many ways to collect the data so we will start with the first one that is working with the csv files, other is sql or json data
whenever u bring data thru the api it is mostly in the json format , then is fetching data from the api , the last is web scrapping
csv ==> comma seprated values

CSV FILE AND READ_CSV parameters ==>

2. Handling "Non-Standard" CSVs
In the real world, you will encounter files that aren't separated by commas or have extra junk at the top.


THESE ARE SOME COMMON PARAMETERS U CAN PASSS INSIDE THE READ_CSV

NAMES ==> YOU CAN PASS THE LIST OF NAMES USING THE NAME PARAMTER IT WILL ASSIGN THE SAME NAMES TO ALL THE COLUMNS ACCORDING TO THE LIST
THIS MOSTLY HAPPENS IN CASE WHERE THERE IS NO HEADING GIVEN TO THE COLUMNS AND THE FIRST ROW OF THE DATA IS SET AS THE COLUMN HEADING
df = pd.read_csv('movie_metadata.tsv",name==["hatt","kutee","oasjg"]

HEADER ==> NOW IN CASE WEHRE THE ACTUAL HEADING OR THE COOLUMMN NAME IS INCLUDED IN DATA AS THE I.E LIKE THE 0TH  ROW IS THE COLUMN HEADINGS ONT THE
ACTUAL DATA IN THAT CASE U CAN USE THE PARAMETER KNOWN AS THE HEADER JUST PASS THE HEADER AS THE 1
df = pd.read_csv('movie_metadata.tsv",header=1)
USE_COLUMNS ==> SOMETIMES U JJUST NEED SOME COLUMNS IN THE DATASET NOT ALL OF THEM SO U CAN PASS THE LIST OF THE COLUMNS U NEED 
INSIDE THIS PARAMETER
df = pd.read_csv('movie_metadata.tsv,columns=["gender","age","salary"]
NROWS ==> IF U NEED ONLY FIRST 100 ROWS U CAN PASS THE VALUE OF NROWS = 100 
Custom Separators: If your file uses tabs or semicolons.
THIS SEP IS USED TO TELL THE PANDAS ON WHICH THING YOU HAVE TO SEPRATE THE VALUES DEFAULT IS SET ON THE COMMAS U CAN OVERWRITE IT
df = pd.read_csv('movie_metadata.tsv', sep='\t')    THIS WILL BE USED FOR TAB SEPRATED FILES as \t stands for  tab

DTYPES ==> U CAN USE THE DTYPES TOO CHANGE THE DATATYPE OF ANY COLUMN LIKE IF ITS A FLOAT U CAN CONVERT IT INTO THE INTEGER
df = pd.read_csv('movie_metadata.tsv",dtyp={"column_name":datatype}#so if u want to convert it into a string u can write string inplace of datatype

PARSE_DATES ==> NORMALLY THE DATES WRITTEN IN THE DATASET ARE IN THE STRING FORMAT BUT IF U WANT TO USE THEM AS A DATE ONLY U CAN USE
THE PARSE DATE AS THE PARAMETER AND PASS THE NAME OF THE COLUMNS THAT ARE REALLY THE DATES
f = pd.read_csv('movie_metadata.tsv",parse_dates=["date_match"] here date_match would be the column which is a date but is in the string format 

Skipping Rows: Sometimes the first few lines of a CSV are just notes or descriptions from the person who created it.
df = pd.read_csv('data.csv', skiprows=2) # Skips the first 2 rows #U CAN ASLO USE A FUNCTION OF LAMBDA LIKE TO GET ONLY EVEN ROWS 


Encoding Issues: If you get a UnicodeDecodeError, it's usually because the file has special characters. Use latin-1 or utf-8
df = pd.read_csv('global_data.csv', encoding='latin-1')

3. Memory Management (The "Big Data" Trick)
If your dataset is massive (millions of rows), your computer might crash if you try to load it all at once. This is where Chunking comes in—a concept you'll need if you ever want to work as a Data Engineer.

The chunksize parameter: Instead of a full DataFrame, this returns an object you can loop through.

# Load 1,000 rows at a time
WHEREEVER THE DATA SET IS VEERY LARGE U JUST USE THE CHUNKSIZE PARAMETER AND THEN PERFORM THE FUNCITONS ON THE DATA
IN A FOR LOOP FOR EACH CHUNK SIZE
chunks = pd.read_csv('large_data.csv', chunksize=1000)

for chunk in chunks:
    # Process each 1,000-row piece individually
    print(chunk.shape)
'''