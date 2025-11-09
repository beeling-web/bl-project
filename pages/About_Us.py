import streamlit as st

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="My Streamlit App"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("About Contract Assistant App")

st.write ("This application provides help you in reading the contract documents in short period of time. There are three different ways to help you quickly find and understand crucial information.")
st.write ("1. Main: The interactive chatbot allows you to ask questions and receive direct answers based solely on the loaded contract documents.\n" \
             "2. Extract Clause: This page processes your request to isolate and display specific contract clauses (like SLA, Termination and etc) from the document, enabling targeted review.\n" \
              "3. Summary Contract: This page generates a brief, concise summary of any specific section or the whole contract, helping you quickly grasp the essential terms.")


with st.expander("How to use this App"):
    st.write("1. Enter your prompt in the text area/ select the option.")
    st.write("2. Click the 'Submit' button.")
    st.write("3. The app will generate a text completion based on your prompt.")
