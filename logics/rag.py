##referred to week 4 materials
from langchain_community.document_loaders import PyPDFLoader # for document loading
from langchain_text_splitters import RecursiveCharacterTextSplitter #For Splitting & Chunking
from helper_functions.llm import count_tokens
from langchain_chroma import Chroma #for Vectorstores
from langchain_openai import OpenAIEmbeddings # for Embedding 
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough  #LangChain Expression Language (LCEL) approach
from langchain_core.output_parsers import StrOutputParser #LangChain Expression Language (LCEL) approach




##Documents Loading (ICT contract PDFs DOCX) using Document Loaders (e.g., PyPDFLoader, DirectoryLoader)

loader = PyPDFLoader("./data/IT SYSTEM MAINTENANCE AND SERVICE LEVEL AGREEMENT.pdf")
pages = loader.load()



##Splitting & Chunking (using RecursiveCharacterTextSplitter)
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=500,
    chunk_overlap=50,
    length_function=count_tokens
)

splitted_documents = text_splitter.split_documents(pages)

#for testing and check the number of doc it splitting to - commented after successly done
'''
for doc in splitted_documents:
    print(count_tokens(doc.page_content))

'''

#for testing- commented after successly done
'''
print(splitted_documents[4])
'''



##Storage: Embedding & Vectorstores using OpenAi Embedding

# The specified model is 'text-embedding-3-small'.
embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')

vector_store = Chroma.from_documents(
    collection_name="contracts",
    documents=splitted_documents,
    embedding=embeddings_model,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
)

##testing the number of vector_store created - commented after successly done
'''
print (vector_store._collection.count())
'''


# How to load the vector store from the disk - commented after using template approach
'''
vector_store = Chroma("contracts",
                      embedding_function=embeddings_model,
                      persist_directory= "./chroma_langchain_db")
'''



## Basic Retrival - commented after using template approach
'''
vector_store.similarity_search('LD', k=3)
'''
# 1. Define the components (Assuming QA_CHAIN_PROMPT is defined)
# Build prompt


template = """You are contract analyst. Use the following pieces of context to answer the question. Understand the important contract details (like date, parties, supplier, customer and duration) and do not miss out them. If you don't know the answer, just say that you don't know, don't try to make up an answer. Always say "\nThanks for asking!" at the end of the answer.
Contract: {context}
Question: {question}
Answer:"""


extract_template = """You are specialist in extracting the contract clauses. Use the following pieces of context and identify the relevant sections and paragraphs from the context. If it is in table format, display the whole table. Completeness is important. If you don't know the answer, just say that information could not be found, don't try to make up an answer.
Contract: {context}
Question: {question}
Output your response in the following format:
Contract Titile: \n
Extracted clause(s): \n 
Include the Article Number and its Description, and followed by all sub points related to {question}. If there is a table in the contract, display the table. Completeness is important. Do not miss out important information. 

"""

summary_template = """You are good at summarising contract. Use the following pieces of context to provide summary based on the question. Use five sentences maximum. Keep the answer as concise in point forms.
Contract: {context}
Summary approach: {question}
Output your response in the following format:
Contract Title: \n
Summary: \n 
The summary of based on {question}.

"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

EXTRACT_CHAIN_PROMPT = PromptTemplate.from_template(extract_template)

SUMMARY_CHAIN_PROMPT = PromptTemplate.from_template(summary_template)

##  RetrievalQA is deprecated
# changed it to LangChain Expression Language (LCEL) approach, which is the current standard.


rag_chain = (
    {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
    | QA_CHAIN_PROMPT
    | ChatOpenAI(model='gpt-4o-mini')
    | StrOutputParser()
)

rag_extract_chain = (
    {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
    | EXTRACT_CHAIN_PROMPT
    | ChatOpenAI(model='gpt-4o-mini')
    | StrOutputParser()
)

rag_summary_chain = (
    {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
    | SUMMARY_CHAIN_PROMPT
    | ChatOpenAI(model='gpt-4o-mini')
    | StrOutputParser()
)

#- the followin are commented after using template approach
#LD clause extraction
#result = rag_chain.invoke("Extract LD and penalties related clause, including Uptime Breach Penalty, P1 Resolution breaach and LD Cap")

#SLA clause extraction
#result = rag_chain.invoke("Extract SLA clause, including table that include definition, response and resoluton time")

#Termination clause extraction
#result = rag_chain.invoke("Extract Termination clause, including ifferent type of termination and notice period, what is the obligation upon termination")

#print("Final Answer:\n", result)


