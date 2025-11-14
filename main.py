# Set up and run this Streamlit App
import streamlit as st
import pandas as pd
# from helper_functions import llm
from logics import rag
from logics.rag import rag_chain 
from logics.chat_history_handler import save_content
from logics.chat_history_handler import read_and_compare
from helper_functions.utility import check_password  



# 1. Session State Initialization 
# Initialise a session called "messages" to store the previous chat records
if "messages" not in st.session_state:
    st.session_state["messages"] = []



# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="wide",
    page_title="My Streamlit App"
)
# endregion <--------- Streamlit App Configuration --------->

# Check if the password is correct.  
if not check_password():  
    st.stop()


st.header("My Contract Assistant App")


# 2. Initialise a container called "response_container" to display AI response
response_container = st.container(height=300)

# 3. Defines response_container's content
with response_container:

    # Encourage user to start conversation if session "messages" is empty (no past conversation in this web session)     
    if not st.session_state["messages"]:
        st.info("Start a conversation by asking me the first question.")
    
    #  Display all messages from session "messages" 
    for message in st.session_state["messages"]:
        role = message["role"]
        content = message["content"]
        
        # Use st.chat_message for a native chat look
        with st.chat_message(role):
            st.markdown(content)


#  --- Form Definition ---
# Defines the form and its content.
with st.form(key="form"):
    
    # Text area for user input
    user_prompt = st.text_area("What do you want to know about our contracts?", height=90)
    
    # Submit button
    submitted = st.form_submit_button("Submit")


with st.expander ("Disclaimer"):
    st.write ("IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.")
    st.write("Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.")
    st.write ("Always consult with qualified professionals for accurate and personalized advice.")


# --- Submission Logic ---
# This code runs after the form is submitted.
if submitted and user_prompt:

    st.toast(f"User Input Submitted - {user_prompt}")

    # 1. Store the prompt and add User message to history
    st.session_state["messages"].append({"role": "user", "content": user_prompt})


    #2. Compare the text file that stores all the chat history and return the previous response from the file
    history_chat= read_and_compare ("data/chathistory_user1.txt", user_prompt)
    
    #3. Invoke your RAG chain when required

    #if there is no chat history stored in the text file, invoke RAG
    if len(history_chat)==0:
        return_fr_rag = rag_chain.invoke(user_prompt)
        response = f"**AI Response to the question:** '{user_prompt}'\n\n {return_fr_rag}"
    
    #if there is previous reply found stored in the text file, retrieve previous reply from text file
    else: 
        response = f"**AI Response to the question:** '{user_prompt}'\n\n {history_chat}"
    
    # Save the response back to the text file
    save_content("data/chathistory_user1.txt",response)
    
    # 3. Add the AI's response to history
    st.session_state["messages"].append({"role": "assistant", "content": response})

    # 4. Force the app to immediately re-run and show the new history
    st.rerun()


