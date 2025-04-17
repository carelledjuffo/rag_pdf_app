from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
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
from flashrank import Ranker
from langchain.retrievers.document_compressors import FlashrankRerank
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
        loader = PDFPlumberLoader(file_path)
        pages = loader.load()  # Each page has its own metadata['page']
        print(pages[0].page_content)
        return pages
    except Exception as e:
        logging.error(f"Failed to extract PDF: {e}")
        return []

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
        documents = [Document(page_content=doc.text) for doc in llama_parse_documents]
        save_documents_to_markdown(documents)

        return documents
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return []
def clean_pdf():
    return 0

def create_chunks(pages):
    #splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    splitter = SemanticChunker(OpenAIEmbeddings())
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
        retriever = vector_store.as_retriever()

        retriever_from_llm = MultiQueryRetriever.from_llm(
         retriever=retriever, llm=llm
        )

       # ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

        #FlashrankRerank.RANKER = ranker
        
        #FlashrankRerank.model_rebuild()

        
        #compressor = FlashrankRerank()

        #compression_retriever = ContextualCompressionRetriever(
        #    base_compressor=compressor, base_retriever=retriever
        #)

        return retriever_from_llm
    except Exception as e:
        logging.error(f"Failed to create retriever: {e}")
        return "", {}

    
def split_data(documents):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        print(f"Parsed documents: {len(documents)}")
        print(f"Chunks created: {len(chunks)}")

        return chunks
    except Exception as e:
        logging.error(f"❌ Failed to split data: {e}")
        return []
    

def split_data_semantic(documents):
    try:
        text_splitter = SemanticChunker(OpenAIEmbeddings())
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



def qa_chain_process_llama_parse(documents):
    chunks = split_data_semantic(documents)
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
 


def qa_chain_process_simple(documents):
    chunks = create_chunks(documents)  
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
