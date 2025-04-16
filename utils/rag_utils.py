from langchain_community.document_loaders import PyPDFLoader
#from langchain_chroma import Chroma
#from langchain_community.vectorstores import Chroma
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from llama_parse import LlamaParse
import getpass
import os
import shutil # Importing shutil module for high-level file operations
import logging
import re
import joblib


open_ai_api_key = os.getenv("OPENAI_API_KEY")
llama_parse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")



llm = ChatOpenAI(api_key= open_ai_api_key, model="gpt-4o-mini", temperature=0)
CHROMA_PATH = "chroma_db"

def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()  # Each page has its own metadata['page']
        return pages
    except Exception as e:
        logging.error(f"Failed to extract PDF: {e}")
        return "", {}

def clean_pdf():
    return 0

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

def create_vector_store(chunks):
    if os.path.exists(CHROMA_PATH):
           shutil.rmtree(CHROMA_PATH)
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
        vector_store.persist()
    except Exception as e:
        logging.error(f"Failed to create vector store: {e}")
        return "", {}

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    return vector_store
    

def create_retriever(vector_store):
    try:

        retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 10, "fetch_k": 20}
        )
        return retriever
    except Exception as e:
        logging.error(f"Failed to create retriever: {e}")
        return "", {}


def qa_chain_process_2(file_path):
    pages = load_pdf(file_path)
    chunks = create_chunks(pages)  
    vector_store = create_vector_store(chunks)
    retriever = create_retriever(vector_store)

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
    






async def qa_chain_process(file_path):
    parsed_data = await parse_data(file_path) 
    chunks = split_data(parsed_data)
    vector_store = create_vector_store(chunks)
    retriever = create_retriever(vector_store)

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
 


async def parse_data(file_path):
    data_file = "data/parsed_data.pkl"
    try:
        # Perform the parsing step and store the result in llama_parse_documents
        parsingInstructionAcademicPaper = """
                The provided document is an academic research paper, typically published in a conference proceedings or journal.
                It contains structured sections such as an abstract, introduction, related work, methodology, experiments, results, and conclusion.
                Figures, tables, and equations are often included to support findings.
                Be precise and focus on the core contributions, methods, datasets used, and key results when answering questions.
                Ignore boilerplate text like licensing, template instructions, or repeated headers.
        """

        parser = LlamaParse(
            api_key=llama_parse_api_key,
            result_type="markdown",
            parsing_instruction=parsingInstructionAcademicPaper,
            max_timeout=5000,
        )
        llama_parse_documents = await parser.aload_data(file_path)


        # Save the parsed data to a file
        print("Saving the parse results in .pkl format ..........")
        joblib.dump(llama_parse_documents, data_file)

        # Set the parsed data to the variable
        parsed_data = llama_parse_documents

        return parsed_data
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return "", {}



def split_data(llama_parse_documents):
    try:
        documents = [Document(page_content=doc.text) for doc in llama_parse_documents]
        save_documents_to_markdown(documents)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        print(f"Parsed documents: {len(documents)}")
        print(f"Chunks created: {len(chunks)}")

        return chunks
    except Exception as e:
        logging.error(f"❌ Failed to split data: {e}")
        return []
    

def save_documents_to_markdown(documents, filepath="data/output.md"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for i, doc in enumerate(documents, start=1):
            f.write(f"## Document {i}\n\n")
            f.write(doc.page_content.strip() + "\n\n")