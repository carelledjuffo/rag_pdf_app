from langchain_community.document_loaders import PyPDFLoader
#from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
#from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import getpass
import os
import shutil # Importing shutil module for high-level file operations


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = getpass.getpass("Enter your OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = api_key


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
CHROMA_PATH = "chroma_db"

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()  # Each page has its own metadata['page']
    return pages

def create_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for page in pages:
        page_number = page.metadata.get('page', 'Unknown')  
        split_chunks = splitter.split_documents([page]) 
        for i, chunk in enumerate(split_chunks):
            chunk.metadata['page_number'] = page_number
            chunk.metadata['chunk_number'] = i + 1
            chunk.metadata['section_name'] = f"Section on page {page_number}"
            chunks.append(chunk)
    return chunks

def save_to_chroma_create_retriever(chunks):

    if os.path.exists(CHROMA_PATH):
       shutil.rmtree(CHROMA_PATH)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
    vectorstore.persist()

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

    #retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 10, "fetch_k": 20}
    )
    return retriever

def qa_chain_process_2(pages):

    chunks = create_chunks(pages)  
    retriever = save_to_chroma_create_retriever(chunks)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
            You are an expert assistant. Use exclusively the following context below to answer the question accurately.
            If the answer is not found in the context , let me know and do not invent any answer
            Context:
            #####{context}####

            Question:
           #### {question}###

            Answer:
            """
   )
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
    )
    return {"qa_chain": qa_chain, "retriever":retriever}
    











def qa_chain_process(docs):


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    #print("type of slpits", type(splits[0]))
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    #print("Embeddings", embeddings.embed_query("Hello, world!"))

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="chroma_db")
    retriever = vectorstore.as_retriever()
    
    
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        retriever=retriever,
        #persist_directory="chroma_db"
    )

    return qa_chain
 
