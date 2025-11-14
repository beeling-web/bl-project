import streamlit as st

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="wide",
    page_title="My Streamlit App"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("About Contract Assistant App")
st.header ("Methodology")

st.warning ("The flow chart is at the bottom of the page.")
# 4. Using Markdown for 'Sub-Sub-Text' (H4)
# We now use four hashes (####) to keep the hierarchy smaller.
st.markdown("##### This flow handles user's inquiries using a Retrieval-Augmented Generation (RAG) approach.")

st.markdown(
"""
**1. Preparing knowledge base:**

a. Input: The Contract document, which is stored at 'data' folder.\n
b. Processing: Load, split and chunk the contract document, breaking it into smaller text segments.\n
c. Output: These text chunks are then converted into numerical vectors (embeddings) using an embedding model and stored in the Vector Store. 
"""
)



st.markdown(
    """
**2. Main Contract Chatbot:**

a. Input: Users' query. \n
b. Flow: Checks if the query asked previously. If yes, it retrieves previous responses from text file and displays it to user directly.\
    If no, it retrieves the relevant part of the vector store and passed to the LLM to enhance the response.\n
c. Technique used in prompting: ***Chain of Verification (CoVe)*** and ***Sandwich Defence*** \n
d. Output: User responses was appended at the response containers. Text file updated with the new responses.\n

    """
)


st.markdown(
    """
**3. Dedicated Tools: Clause Extraction**

a.Input: Receives a specific users' option (e.g., termination).\n
b. Flow: This option is used to identify and retrieve the relevant part of the vector document (potentially multiple chunks) . The LLM then enhances response by \
    extracting and formatting the clauses into response. \n
c. Technique used in prompting: ***Chain of Thought (CoT)*** and ***Sandwich Defence***.\n
d. Output: Displaying responses to user.

    """
)


st.markdown(
    """
**4. Dedicated Tools: Contract Summary Tool**

a. Input: Receives a general or specific users' prompt on how they want to summarise the contract. (e.g., 'Summarize customer responsibilities \n
b. Flow: It Retrieves the relevant part of the vector document necessary for the summary. The LLM then performs the complex \
task of summarization and enhances response.\n
c. Technique used in prompting: ***Chain of Density (CoD)*** and ***Sandwich Defence***.\n
d. Output: Displaying responses to user.\n
    """
)


st.markdown("---") # Horizontal line for visual separation

st.image("data/flowcharts.jpg", caption="My Methodology Flowchart", width=800)

with st.expander ("Disclaimer"):
    st.write ("IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.")
    st.write("Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.")
    st.write ("Always consult with qualified professionals for accurate and personalized advice.")

