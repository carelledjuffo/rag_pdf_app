import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import gradio as gr
from fastapi.middleware.wsgi import WSGIMiddleware
from gradio.routes import mount_gradio_app
from PyPDF2 import PdfReader
import shutil
from utils.rag_utils import load_pdf, qa_chain_process_llama_parse, qa_chain_process_simple, parse_data
from test_rag import eval_retrieval_precision




app = FastAPI()


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global documents_simple, documents_llama_parse
    documents_llama_parse = None 
    try:
        file_location = f"data/uploaded_pdfs/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            documents_simple = load_pdf(file_location) 
        return JSONResponse(content={"message": f"File '{file.filename}' uploaded successfully!"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/upload-pdf_llama_parse/")
async def upload_pdf_llama_parse(file: UploadFile = File(...)):
    global documents_llama_parse, documents_simple
    documents_simple = None
    try:
        file_location = f"data/uploaded_pdfs/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            documents_llama_parse = await parse_data(file_location) 
        return JSONResponse(content={"message": f"File '{file.filename}' uploaded successfully!"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/ask/")
async def ask_question(query: str):
    global qa_chain_result
    global result
    if not documents_simple and not documents_llama_parse:
        return {"error": "No PDF uploaded yet."}
    if documents_simple is not None:
        print("########## using simple doc upload")
        qa_chain_result =  qa_chain_process_simple(documents_simple) 
    elif documents_llama_parse is not None: 
         print("########## using llama_parse doc upload")
         qa_chain_result = qa_chain_process_llama_parse(documents_llama_parse)  
    result = qa_chain_result["qa_chain"].run(query)
    return {"answer": result}


@app.post("/explain/")
async def explain_rag(query: str):
    if not qa_chain_result:
        return {"error": "No PDF has been processed yet."}
    relevant_docs = qa_chain_result["retriever"].get_relevant_documents(query)
    
    for doc in relevant_docs:
        print(f"- Page Number: {doc.metadata.get('page_number', 'Unknown')}")
        print(f"  Chunk Number: {doc.metadata.get('chunk_number', 'Unknown')}")
        print(f"  Section Name: {doc.metadata.get('section_name', 'Unknown')}")
        print(f"  Content Preview: {doc.page_content[:900]}...\n") 
    return {"answer": "XAI works"}

@app.post("/eval/")
async def eval_rag(query: str):
    if not qa_chain_result:
        return {"error": "No PDF has been processed yet."}
    if not result:
       return {"error": "No question has been asked yet."} 
    relevant_docs = qa_chain_result["retriever"].get_relevant_documents(query)
    retrieval_context = []
    for doc in relevant_docs:
        retrieval_context.append(doc.page_content)
    expected_output = "Ayman Asad Khan, Md Toufique Hasan, Kai Kristian Kemell, Jussi Rasku, Pekka Abrahamsson"
    eval_retrieval_precision(query, result, expected_output, retrieval_context)
    return {f"answer": "Eval works"}