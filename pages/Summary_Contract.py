# Set up and run this Streamlit App
import streamlit as st
import pandas as pd
# from helper_functions import llm
from logics import rag
from logics.rag import rag_summary_chain 
from helper_functions.utility import check_password  


# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="My Streamlit App"
)
# endregion <--------- Streamlit App Configuration --------->



# Check if the password is correct.  
if not check_password():  
    st.stop()


st.title("My Contract Assistant App")

form = st.form(key="form")
form.header("Your Contract Summary Tool")
#form.subheader("What do you want to extract?")


user_prompt = form.text_area("How do you want me to summary the contract?", height=200)

with st.expander ("Disclaimer"):
    st.write ("IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.")
    st.write("Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.")
    st.write ("Always consult with qualified professionals for accurate and personalized advice.")

if form.form_submit_button("Submit"):


    st.toast(f"User Input Submitted - {user_prompt}")

    st.divider()
    
    response = rag_summary_chain.invoke(user_prompt)


    #response = process_user_message(user_prompt)
    st.write(response)

    st.divider()

 
