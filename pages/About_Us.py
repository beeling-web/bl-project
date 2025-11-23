import streamlit as st

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="My Streamlit App"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("About Contract Assistant App")

st.write ("This application helps you in digesting the contract documents in short period of time. There are three different ways to help you quickly find and understand crucial information.")
st.write ("1. Main: The interactive chatbot allows you to ask questions and receive direct answers based solely on the loaded contract documents.\n" \
             "2. Extract Clause: This page processes your request to isolate and display specific contract clauses (like SLA, Termination and etc) from the document, enabling targeted review.\n" \
              "3. Summary Contract: This page generates a brief, concise summary of any specific section or the whole contract, helping you quickly grasp the essential terms.")


st.markdown(
    """
    This project implements a **Retrieval-Augmented Generation (RAG)** architecture to transform contract analysis, 
    offering speed, and efficiency for contract documents review.
    """
)
st.markdown("---")


# --- SECTION 1: CORE OBJECTIVE & SCOPE ---
st.header("1. 🚀 Core Objective & Scope")

st.markdown("""
    | Category | Description |
    | :--- | :--- |
    | **Objective** | To significantly **reduce review time** by offering high-accuracy **AI-powered Q&A, extraction, and summarization** based purely on the provided contract text. |
    | **Scope** | Development of a robust RAG and three key user features for efficient contract understanding. |
    | **Data Source** | **Contract Documents** which is **chunked, embedded, and stored** in the **Vector Store**. |
""")

st.markdown("---")


# --- SECTION 2: KEY FEATURES & FUNCTIONALITY ---
st.header("2. ✨ Key Features & Functionality")
st.markdown(
    """
    All features rely on the **Vector Store** as the single source of truth, ensuring answers are grounded only in the contract text.
    """
)

# Using st.markdown for the table allows for easy copying of the structure
st.markdown("""
    | Feature | Primary Function | Implementation Detail (Data Flow) |
    | :--- | :--- | :--- |
    | **Main Contract Chatbot** | **Contextual Q&A** (e.g., "What is the liability cap?"). | Uses RAG to **retrieve relevant chunks** from the Vector Store, combined with response caching in text files for speed. |
    | **Clause Extractor** | **Clause(s) retrieval** of specific sections (e.g., "Termination"). | Queries the Vector Store and uses the LLM to **extract and format** the exact clause text. |
    | **Contract Summary Tool** | **Synthesizing summaries** (e.g., "Summarise the confidentiality and data security requirement"). | Retrieves necessary document context and uses the LLM for complex **summarisation and synthesis**. |
""")

# --- NEW SECTION 3: HOW TO USE THIS APP ---
st.header("3. 💡 How to Use This App")
st.markdown(
    """
    The application is designed for straightforward interaction across its three main feature tabs:
    """
)

st.markdown(
    """
    1.  **Enter your prompt** in the text area or **select the option** from the dropdown menu (depending on the feature).
    2.  Click the **'Submit' button**. 
    3.  The app will then generate and **display a text completion** based on your prompt and the contract's knowledge base.
    """
)

with st.expander ("Disclaimer"):
    st.write ("IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.")
    st.write("Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.")
    st.write ("Always consult with qualified professionals for accurate and personalized advice.")

