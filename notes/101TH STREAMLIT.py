'''
Welcome to the world of Streamlit. It’s the closest thing to "magic" in the Python ecosystem. It turns a boring Python script into a beautiful, interactive web app in seconds.

Since you already know how to train and save your models, here is the roadmap to becoming a Streamlit pro.
THE STREAM LIT WORKS ON THE DECLARTIVE CODE OF PYTHON WHICH DOESNT REQUIRE ANY OTHER WEB LANGUAGE LIKE CSS/HTML/JAVASCRIPT
What is Streamlit?
Think of Streamlit as a "Bridge." On one side, you have your Python Data Science world (Pandas, XGBoost, CatBoost). 
On the other side, you have the Web (HTML, CSS, Buttons, Sliders).

Normally, you need a whole team of web developers to connect those two. Streamlit allows you to do it using only Python.

The 3 Core Pillars of Streamlit
No Front-End Knowledge Required: You don't need to know what a <div> or a <button> tag is in HTML. You just write st.button("Click Me").

The "Script" Mental Model: Streamlit treats your code like a recipe. Every time a user changes an input (like moving a slider), Streamlit runs your script from top to bottom.
Instant Deployment: It is designed to be pushed to the cloud (GitHub/Streamlit Cloud) in minutes.
'''
#OUTPUT FUNCTIONS OF STREAMLIT
'''
you need to know that Streamlit functions are divided into Inputs (to get data from the user), Outputs (to show results), and Layouts (to make it look like a real website).

Here are the most used functions with the details you need to actually use them in your code.

1. The "Big Three" Output Functions
These are used to display your analysis and model results.

st.write()
The "Swiss Army Knife." It automatically figures out what you are passing it (a string, a dataframe, a chart, or a dictionary) and renders it perfectly.
Details: Best for quick debugging or showing simple text and tables.

Example: st.write("Model Accuracy:", 0.95)

st.metric()
Displays a large, prominent number with an optional "delta" (change) indicator.
Details: Perfect for showing performance metrics like AUC Score or Precision.

Example: st.metric(label="AUC Score", value=0.88, delta=0.02)

st.dataframe()
Displays an interactive table that users can sort, filter, and expand.
Details: Unlike st.table() (which is static), this allows users to explore the data.

Example: st.dataframe(df.head(10))'''

# MAIN INPUT FUNCTIONS OF STREAMLIT
'''
2. The "Feature Input" FunctionsThese are what you use to replace your manual x_test data. You capture these values and put them into a list to feed your model.predict().
st.slider()
A horizontal slider for numerical ranges.

Details: Arguments are (label, min_value, max_value, default_value).

Example: age = st.slider("Select Age", 18, 100, 25)

st.selectbox()
A dropdown menu for categorical data.

Details: Returns the string of the selected option. Great for features like "City" or "Gender".

Example: city = st.selectbox("Choose City", ["New York", "London", "Tokyo"])

st.button()
A clickable button that returns True only in the moment it is clicked.

Details: You wrap your prediction logic inside an if statement with this.

Example: ```python
if st.button("Predict"):
st.write("Result is...")
'''

import streamlit as st
st.title("hello world")#this would give you the title 
st.subheader("made with streamlit")#this is like a subheading
st.text("this is your first thing")#this is like a default one
st.write("this is similar to text")
city=st.selectbox("select the city",["hatt","kutta","chall"])#this is like a drop down menu too select options u want
''' the first thinig u pass in the select box is the name u wanna giive that box the other is the array from which u wanna choose the optin'''
''' u can simply use that wihtout putting in a variable but if u wanna use the selected option further as well u have to use it in the variable'''
st.write(f"u choosed {city}")
st.success("your city has been selected")
'''st.success() is a Status Element used to display a message in a clean, professional-looking green box. It is the standard way
 to tell your user that an operation (like a model prediction or a file upload) finished correctly.

 SIMILAR TO SUCCES YOU CAN USE OTHERS AS WELL LIKE

 Function                   Color                   Best Use Case
st.success()            Green                   """Model Prediction: Low Risk"""
st.error()              Red                   ""Model Prediction: High Risk / Fraud"""
st.warning()            Yellow                  """Missing values detected in your input!"""
st.info()               Blue                    ""The model is currently using XGBoost v2.0."""

'''