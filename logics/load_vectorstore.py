import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Must match the model and collection name used in your build script (rag.py)
COLLECTION_NAME = "contracts"
PERSIST_DIRECTORY = "./chroma_langchain_db"
EMBEDDING_MODEL = 'text-embedding-3-small'

def load_persisted_vector_store():
    # Check if the persisted directory exists
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(
            f"Vector store not found at {PERSIST_DIRECTORY}. "
            "Please ensure you run the build script locally and push the 'chroma_langchain_db' folder."
        )

    embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    # Load the persisted database 
    vector_store = Chroma(
        collection_name=COLLECTION_NAME, # TO FIX for NotFoundError
        embedding_function=embeddings_model,
        persist_directory=PERSIST_DIRECTORY
    )
    
    return vector_store