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
    chunk_size=350,
    chunk_overlap=35,
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


###template = """You are contract analyst. Use the following pieces of context to answer the question. Understand the important contract details (like date, parties, supplier, customer and duration) and do not miss out them. If you don't know the answer, just say that you don't know, don't try to make up an answer. Always say "\nThanks for asking!" at the end of the answer.
###Contract: {context}
###Question: {question}
###Answer:"""


template = """
You are customer service agent in contract management team. You are given a contract as your context and you need to understand your contract in details. \
    Your objective is to help users to understand the contract by answering to their questions.  You need to apply the following 4 steps when answering question.\
        If you don't know the answer, just say that "Information cannot be found in the contract", don't try to make up an answer. \
            Always say "\nThanks for asking!" at the end of the answer.
Contract: {context}
Question: {question}

When answering, note that the 'Effective Date' is the same as the 'contract start date', and 'Authority' is the same as 'Customer'.

Step 1: Given the question: "{question}" Create an initial response.
Step 2: Based on the question "{question}" and the initial response, generate a list of 1-2 verification questions to fact-check the response.
Step 3: Answer these verificaton questions independently to verify the fact.
Step 4: Using the original question, the initial response, and the verification answers, generate a final, verified response to the question.

Remember, you need to answer user question regarding to the contract by performing the above 4 steps. Do not show your work. Only show the final and verified response in step 4 to user.
"""

'''
extract_template = """You are specialist in extracting the contract clauses. Use the following pieces of context and identify the relevant sections and paragraphs from the context. If it is in table format, display the whole table. Completeness is important. If you don't know the answer, just say that information could not be found, don't try to make up an answer.
Contract: {context}
Question: {question}
Output your response in the following format:
Contract Titile: \n
Extracted clause(s): \n 
Include the Article Number and its Description, and followed by all sub points related to {question}. If there is a table in the contract, display the table. Completeness is important. Do not miss out important information. 

"""
'''



extract_template = """You are specialist in extracting the contract clauses in contract management team. \
    You are given a contract as context and you need to understand your contract in details. \
        Your objective is to display the relevant clauses of their questions. \
    You need to apply the following two steps in extracting the relevant clause. 

    STRICT FILTERING RULE: Only content directly related to the Article/Section requested in the {question} must be included in the final output. \
        ANY irrelevant information, including sub-points from different sections (e.g., Article 2, Article 4, etc.), MUST be excluded.

Step 1: Identify the relevant article , sections, paragraphs, sub-points and table in the {context} based on {question} ONLY. 
Step 2: Display ONLY the identified article, sections, paragraphs, sub-points and table from the contract. If a clause has sub-points, include them without exception.\
If a clause has a table, include all data points and present them in a simple, itemized list format where each row's data is grouped together.

If you don't know the answer, just say that information could not be found, don't try to make up an answer.

Contract: {context}
Question: {question}
Output your response in the following format:
Contract Title: \n
Extracted clause(s): Include the Article Number and its Description, and followed by all related sub-points (a, b, c, etc.) MUST be included in their entirety. For any table data, display all rows clearly. If a formal table cannot be constructed, use a clear, itemized list format where each row's data is grouped together (e.g., 'P1: System Down, Response: 15 min, Resolution: 4 hrs'). Completeness is important.\n 

Remember, you need to extract all the relevant clauses regarding to the contract by performing the above 2 steps. 

"""


summary_template = """You are contract summary specialist in contract management team. \
    You are given a contract as your context and you need to understand your contract in details. \
    Your objective is to summarise the contract using the approach stated in customer's prompt. \
        You need to apply the following 2 steps when summarising the contract. \
            You will generate increasingly concise, entity-dense summaries of the contract. 

Repeat the following 2 steps 3 times:
Step 1: Identify 1-3 informative entities from the contract which are missing from the previously generated summary.
Step 2: Write a new, denser summary of identical length which covers every entity and details from the previous summary plus missing entities. 

A missing entity is: 
- Relevant: to the contract
- Specific: descriptive yet concise (5 words or fewer)
- Novel: Not in previous summary
- Faithful: present in the contract
- Anywhere: located anywhere in the contract

Guideline: 
- The first summary should be long (4-5 sentences) ~80 words yet highly non-specific, containing little information beyond the entities marked as missing. 
- Make every word count by rewrite the previous summary to improve the flow and make space for additional entities. Make space for fusion, compression and removal of uninformative phrases.
- The summary should become highlighted dense and concise yet self-contained. Can be easily understood without referring to the contract.
- Missing entities can appear anywhere in the new summary, never drop the entities in previous summary. If space cannot be made, add fewer new entities. 
- Use point forms or table whenever applicable.
- If you don't know the answer, just say that information could not be found, don't try to make up an answer.

Contract: {context}
Summary approach: {question}

Output your response in the following format:
Contract Title: \n
Summary: \n 
The summary of based on {question}.

Remember, you need to provide summary regarding to the contract by performing the above 2 steps for 3 times. Do not show your work. Only show the most concise, entity-dense summaries of the contract provided in Step 2 at third round to user.
"""


QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

EXTRACT_CHAIN_PROMPT = PromptTemplate.from_template(extract_template)

SUMMARY_CHAIN_PROMPT = PromptTemplate.from_template(summary_template)


# 1. Define a standard retriever for QA Chain and Summary chain (k=4 is often enough)
qa_summary_retriever = vector_store.as_retriever() # Uses default k (usually 4)

# 2. Define a dedicated retriever for EXTRACTION, using a higher k
# We choose k=8 to ensure all sub-points and surrounding context are included.
extract_retriever = vector_store.as_retriever(
    search_kwargs={"k": 8}
)


##  RetrievalQA is deprecated
# changed it to LangChain Expression Language (LCEL) approach, which is the current standard.


rag_chain = (
    {"context": qa_summary_retriever, "question": RunnablePassthrough()}
    | QA_CHAIN_PROMPT
    | ChatOpenAI(model='gpt-4o-mini')
    | StrOutputParser()
)

rag_extract_chain = (
    {"context": extract_retriever, "question": RunnablePassthrough()}
    | EXTRACT_CHAIN_PROMPT
    | ChatOpenAI(model='gpt-4o-mini')
    | StrOutputParser()
)

rag_summary_chain = (
    {"context": qa_summary_retriever, "question": RunnablePassthrough()}
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


