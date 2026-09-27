'''
The Layout section is where you stop building "scripts" and start building "apps." By default, Streamlit puts everything in one long vertical line.
Layout functions allow you to move widgets to the side, stack them horizontally, or hide them inside tabs
Here are the four pillars of Streamlit layout that you need to master.

1. st.sidebar (The Control Panel)
This is the most used layout feature. It creates a collapsible grey panel on the left. It’s the perfect place for "Input Features" so they don't clutter your main results area
How to use it: Just add .sidebar between st and the widget name.
'''
# EXAMPLE ==>
# # This goes in the main area
# st.title("Model Dashboard")

# # This goes in the sidebar
# age = st.sidebar.slider("Select Age", 0, 100)
# gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
'''2. st.columns() (Horizontal Stacking)
If you have 10 input fields, your app will be 3 miles long. st.columns lets you put them side-by-side.

How to use it: It returns a list of "column objects." You use the with command to put things inside them.'''
# EXAMPLE ==>
# col1, col2, col3 = st.columns(3)

# with col1:
#     st.header("Feature A")
#     val_a = st.number_input("Input A", key="a")

# with col2:
#     st.header("Feature B")
#     val_b = st.number_input("Input B", key="b")

# with col3:
#     st.header("Feature C")
#     val_c = st.number_input("Input C", key="c")
'''
3. st.tabs() (The Organizer)
Tabs are great for separating different stages of your project (e.g., "Data Exploration" vs. "Model Prediction").

How to use it: Pass a list of names to st.tabs().

Example:'''
# # EXAMPLE ==>.
# tab1, tab2 = st.tabs(["📈 Visualization", "🤖 Prediction"])

# with tab1:
#     st.write("Here is the chart of your data.")
#     # st.line_chart(df)

# with tab2:
#     st.write("Enter values to get a prediction.")
#     # if st.button("Predict"): ...
'''4. st.expander() (The "Show More" Box)
Sometimes you have technical details (like Model Hyperparameters or Raw Data) that most users don't need to see immediately.

How to use it: It creates a clickable dropdown box.
'''
# EXAMPLE ==>
# with st.expander("See explanation of the model logic"):
#     st.write("""
#         This model uses an XGBoost classifier trained on 10,000 rows.
#         The primary features used are Age, Income, and Credit Score.
#     """)


####    CODE    #####
import streamlit as st
tab1,tab2=st.tabs(["yeh pehla hai","yeh kyu btaana"])
with tab1:
    st.title("layouts",text_alignment="center")
    # now ill be creating columns using the st.columns(n) n represents the no. of columns u want
    ''' U CAN ALSO A RATIO OF THE COLUMN WIDTH IN A LIST LIKE [1,3,1]''' #TIHS WILL CREATE 3 XOLUMNS OF THE GIVEN RATIO
    col1,col2=st.columns(2)# if we wanna add anything in the column 1 we will use the col1. otherwise col2.
    with col1:
        st.header("pehla column hai yeh")
        #now we will try to add a image in this it is necessay u pass the image with the width otherwise it will cover the whole column just copy the image adress
        st.image("https://imgs.search.brave.com/w67A1vKum4n4kACZjtsXXhNWK66i-sF6-fTaPV1oBL4/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9pLnBp/bmltZy5jb20vb3Jp/Z2luYWxzLzE3L2M4/L2UzLzE3YzhlM2M4/NWNhMDExOTRiYjI0/YmVhYjc1M2UxYWU3/LmpwZw"
                ,width=1000)
        if st.button("choose this if u are human"):
            st.success("YES! NOW U CAN DIE")
    with col2:
        st.header("dusra colummn hai yeh")
        st.image("https://imgs.search.brave.com/a1LdJJih7AD0pC__0VD7lFVq2NYlXuq3ve2ZrmiZLbg/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9zdGF0/aWMud2lraWEubm9j/b29raWUubmV0L3Rl/a2tlbi9pbWFnZXMv/Yi9iYi9KaW5fYW5k/X0thenV5YV9UZWtr/ZW5fOF9Qcm9tb3Rp/b25hbC5qcGcvcmV2/aXNpb24vbGF0ZXN0/L3NjYWxlLXRvLXdp/ZHRoLWRvd24vNjcw/P2NiPTIwMjMxMDA1/MjM0NjEyJnBhdGgt/cHJlZml4PWVu"
                ,width=1000)
        if st.button("choose this if u are dog"):
            st.success("YES! NOW U CAN livE")
    # we can also add the side bar with this all to add the certain things u want
    with st.sidebar:
        name=st.text_input("enter your name bacha")
        gender=st.radio("enter you gender",["MALE","FEMALE","KYU PUCHA"])
        age=st.number_input("ENTER YOUR AGE",max_value=100,min_value=0,value=10,step=5)#value is the default value # step is the step size by how much clicking the + icon the age will inc
        if age<=18:
            st.error("CHAL SALE MINOR")
    st.divider()#this will add a horizontal line
    st.info(f"🕵️ **Suspect Profile:** {name} | {gender} | Age: {age}")#THIS CREATES A GOOD INFO THING INSIDE A BLUE BOX


    ''' NOW FOR FORMATTING COLOURING AND ALL THE HTML THINGS WE USE THE MARKDOWN FUCNTION IN ST'''
    '''Think of st.markdown() as the "God Mode" of Streamlit. While functions like st.text() or st.write() are easy to use, they are 
    very rigid. Markdown allows you to use formatting, colors, emojis, links, and even raw HTML to make your app look like a real website.

    Since you are struggling with centering and layouts, st.markdown is actually your best friend.


    1. The Basics (Bold, Italics, Lists)
    Streamlit uses the standard GitHub-flavored Markdown.

    Bold: **Text**
    Italics: *Text*
    Headers: # for Huge, ## for Medium, ### for Small.
    Dividers: --- (Same as st.divider())
    '''
    import streamlit as st

    st.markdown("# 🚀 Main Title")
    st.markdown("This is a **bold** statement for your ML model.")
    st.markdown("1. First Step: Load Data\n2. Second Step: Train XGBoost")
    '''2. The "HTML Hack" (The Secret Sauce)
    This is what you needed for your centering issue. By default, Streamlit treats Markdown as plain text for security. 
    If you want to use CSS (colors, alignment, fonts), you must add the parameter unsafe_allow_html=True

    '''