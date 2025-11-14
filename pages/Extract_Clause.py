# Set up and run this Streamlit App
import streamlit as st
import pandas as pd
# from helper_functions import llm
from logics import rag
from logics.rag import rag_chain, rag_extract_chain
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


#st.title("My Contract Assistant App")

form = st.form(key="form")
##form.header("Your Clause Extracter")
##form.subheader("What do you want to extract?")
form.header ("My Contract Assistant App")
form.subheader("Your Clause Extracter")
##form.markdown("What do you want to extract?")

option_map = {
    0: "LD and Penalties",
    1: "Termination",
    2: "Service Level Agreement (SLA)",
    3: "Supplier Obligation",
}
selection = form.pills(
    "What do you want to extract?",
    options=option_map.keys(),
    format_func=lambda option: option_map[option],
    selection_mode="single",
)

with st.expander ("Disclaimer"):
    st.write ("IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.")
    st.write("Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.")
    st.write ("Always consult with qualified professionals for accurate and personalized advice.")


if form.form_submit_button("Submit"):


    st.toast(f"User Input Submitted - {option_map[selection]}")

    st.divider()

    st.write(
    "Your selected option: "
    f"{None if selection is None else option_map[selection]}"
    )

    response = rag_extract_chain.invoke(option_map[selection])


    
    st.write(response)

    st.divider()