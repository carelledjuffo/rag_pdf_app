import os
import fitz  # PyMuPDF
import logging
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader


# Setup logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_pdf_text(file_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()  # Each page has its own metadata['page']
        return pages
    except Exception as e:
        logging.error(f"Failed to extract PDF: {e}")
        return "", {}

import fitz  # PyMuPDF


def clean_text(text):
    # Remove headers, footers, page numbers, and normalize text
    cleaned = re.sub(r'\n{2,}', '\n', text)  # collapse multiple newlines
    cleaned = re.sub(r'Page \d+', '', cleaned)  # remove page numbers
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)  # collapse multiple spaces
    cleaned = cleaned.strip().lower()
    return cleaned


def chunk_text(pages, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)


def create_documents(chunks, metadata):
    filtered_metadata = {k: v for k, v in metadata.items() if v is not None}
    return [Document(page_content=chunk.page_content, metadata=filtered_metadata) for chunk in chunks]


def build_vectorstore(documents, persist_directory="chroma_db"):
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_directory)
        vectorstore.persist()
        logging.info("Chroma vectorstore created and persisted.")
        return vectorstore
    except Exception as e:
        logging.error(f"Error creating vectorstore: {e}")
        raise


def create_rag_chain(vectorstore):

    docs = vectorstore.similarity_search("What is the main topic of the document?", k=3)
    for i, doc in enumerate(docs):
        print("######################################")
        print(f"\n[Doc {i+1}]\n{doc.page_content[:500]}...")
        print("######################################")
        

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
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


def process_pdf_for_rag(file_path):
    pages = extract_pdf_text(file_path)
    if not pages:
        raise ValueError("No text extracted from PDF.")

    #cleaned_text = clean_text(text)
    chunks = chunk_text(pages)
    #documents = create_documents(chunks, metadata)
    vectorstore = build_vectorstore(chunks)
    qa_chain = create_rag_chain(vectorstore)
    return qa_chain


# Example usage:
if __name__ == "__main__":
    pdf_path = "data/small_task.pdf"  # Replace with your PDF file
    try:
        qa = process_pdf_for_rag(pdf_path)
        query = "What is the main topic of the document?"
        response = qa.run(query)
        print("Answer:", response)
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
