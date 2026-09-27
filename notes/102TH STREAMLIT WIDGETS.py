'''
In Streamlit, Widgets are the interactive elements that allow users to talk to your Python code. Instead of hardcoding values like age = 25,
you use a widget so the user can pick the value themselves.

Every time a user interacts with a widget, Streamlit reruns your entire script from top to bottom, and the variable assigned to that widget is updated with the new value.

1. Selection Widgets (Categorical Data)
These are perfect for features that have specific categories (like "Gender," "City," or "Education Level").

Widget                    Usage                                                 Return Value
st.selectbox()          A dropdown menu for picking one option.                 The selected string.
st.multiselect()        Allows users to pick multiple options.                  A list of strings.
st.radio()              Circular buttons where only one can be picked.          The selected string.
st.checkbox()           A simple on/off toggle.                                  True or False.

2. Numeric & Date Widgets (Continuous Data)
These are essential for numerical features in your Machine Learning models.

st.slider(): Great for ranges.
Syntax: val = st.slider("Label", min, max, default_step)
Example: age = st.slider("Age", 0, 100, 25)

st.number_input(): For when you need exact precision (like "Income" or "Price").
Example: income = st.number_input("Annual Income", min_value=0)

st.date_input(): Displays a calendar.
Example: birthday = st.date_input("Select Date")

3. Text & Media Widgets
st.text_input(): For names, addresses, or single-line strings.
st.file_uploader(): This is a game-changer. It allows users to upload their own CSV or images for your model to process.
Example: file = st.file_uploader("Upload your dataset", type=["csv"])

4. The "Action" Widget: st.button()
This is the most important widget for ML deployment. Since Streamlit reruns every time a slider moves, 
your model might try to predict 100 times while a user is still adjusting their inputs.

The Hack: Wrap your model logic in a button so it only runs when the user is ready.
'''
# import streamlit as st

# # 1. Collect inputs
# age = st.slider("Age", 18, 100)
# smoke = st.checkbox("Do you smoke?")

# # 2. Only run prediction when button is clicked
# if st.button("Predict Health Score"):
#     # (Your model.predict logic goes here)
#     st.success("Analysis Complete!")
'''Pro Tip: The key Parameter
If you have two identical sliders (e.g., "Select Value" for two different features), Streamlit will throw an error. You must give them unique keys:
st.slider("Select Value", 0, 10, key="slider_one")
st.slider("Select Value", 0, 10, key="slider_two")
'''
import streamlit as st
st.title("this is my new file")
if st.button("predict the price"):#its like a clickable button if u will click the next process will start after clicking that only
    st.write("m starting to predict that")
chck=st.checkbox("thiis is to check u are robot or not")#IT IS LIKE A CHECK BOX IF U CEHCK THIS THIS WILL HAPPEN
if chck:#now we will check if this is checked or not
    st.success("YOU ARE A HUMAN U CAN DIE")
choose=st.radio("TELL YOUR GENDER",["MALE","FEMALE","MAI KYU BTAU"])#THIS ISLIKE A OPTION U CAN TICK ONLY ONE
if choose=="MAI KYU BTAU":
    st.error("TOH BHA JHA YHAAN SE KUTTE")
study=st.selectbox("kya kya krlea hai pdhia me abhi tk",["kuch nhi vehla hooon","10vi ki hai","kya krega puch k"])#this is a dropdown menu
if study=="kuch nhi vehla hooon":
    st.text("toh sale aya kyun hai yha pe")
st.slider("kitna mardana hai agr hai",min_value=0,max_value=100,step=5)#it is like a amount of thing how much u want from a numerical coluumn u can select that by sliding
'''
st.slider(): Great for ranges.
Syntax: val = st.slider("Label", min, max, default_step(i.e the default value))
Example: age = st.slider("Age", 0, 100, 25)'''
st.number_input(label="kinee bnde hai tere sath",max_value=100,min_value=0,step=3)#this will take a ni. as the input in the given range
name=st.text_input("naam kya hai tera")
if name:
    st.write(f"kaisa hai re{name}")
#now to entr the date kkind of thing
'''2. The Important Parameters
label: The text the user sees above the calendar (e.g., "Select Start Date").

value: The date selected by default when the app first loads. It defaults to "today".

min_value & max_value: These are crucial for ML apps. They prevent users from selecting dates where you have no data (e.g., if your dataset starts in 2020, set min_value to 2020).

format: You can change how the date is displayed using strings like "DD/MM/YYYY" or "MM-DD-YYYY".    '''
# 1. Set your limits using datetime.date(Year, Month, Day)
import datetime
min_d = datetime.date(2012, 1, 20)
max_d = datetime.date(2026, 12, 31)
default_d = datetime.date(2012, 1, 20)

# 2. Use them in the widget
dob = st.date_input(
    "kab paida hua tha re",
    value=default_d,
    min_value=min_d,
    max_value=max_d
)

st.write("Tera birthday hai:", dob)
'''In Streamlit, st.date_input() is the widget you use to let users select a date from a clean, interactive calendar. In Machine 
Learning, this is commonly used for time-series forecasting or filtering datasets by a specific timeframe.'''
