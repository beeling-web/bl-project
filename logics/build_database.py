import os
# Imports for Document Handling
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
# from helper_functions.llm import count_tokens # Assuming this utility function is available
import tiktoken
from dotenv import load_dotenv


def count_tokens(text):
    encoding = tiktoken.encoding_for_model('gpt-4o-mini')
    return len(encoding.encode(text))

if load_dotenv('.env'):
    # for local development
    OPENAI_KEY =load_dotenv('.env')
else:
    OPENAI_KEY = st.secrets['OPENAI_API_KEY']



# --- Configuration (MUST match load_vectorstore.py) ---
COLLECTION_NAME = "contracts"
PERSIST_DIRECTORY = "./chroma_langchain_db" # Path is relative to the directory from which the script is EXECUTED
EMBEDDING_MODEL = 'text-embedding-3-small'
DOCUMENT_PATH = "./data/IT SYSTEM MAINTENANCE AND SERVICE LEVEL AGREEMENT.pdf"

# --- 1. Document Loading ---
print("--- Starting Database Build ---")
if not os.path.exists(DOCUMENT_PATH):
    print(f"Error: Document not found at {DOCUMENT_PATH}")
    exit()

print(f"1. Loading document from: {DOCUMENT_PATH}")
loader = PyPDFLoader(DOCUMENT_PATH)
pages = loader.load()


# --- 2. Splitting & Chunking ---
print("2. Splitting and chunking documents...")
# Configure your text splitting parameters here:
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=350,
    chunk_overlap=35,
    length_function=count_tokens
)

splitted_documents = text_splitter.split_documents(pages)
print(f"Total chunks created: {len(splitted_documents)}")


# --- 3. Embedding & Vector Store Creation (The Build) ---
print(f"3. Creating embeddings using model: {EMBEDDING_MODEL}")
embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

# Use Chroma.from_documents to build the database from scratch and persist it
vector_store = Chroma.from_documents(
    collection_name=COLLECTION_NAME,
    documents=splitted_documents,
    embedding=embeddings_model,
    persist_directory=PERSIST_DIRECTORY,
)