import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import gradio as gr
import gradio as gr
from fastapi.middleware.wsgi import WSGIMiddleware
from gradio.routes import mount_gradio_app
from PyPDF2 import PdfReader
import shutil
from utils.rag_utils import load_pdf, qa_chain_process, qa_chain_process_2

app = FastAPI()


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global qa_chain_result
    global retriever
    try:
        # Save the uploaded file to disk (you can modify this to perform any processing)
        file_location = f"data/uploaded_pdfs/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            docs = load_pdf(file_location)
            qa_chain_result = qa_chain_process_2(docs) 

        
        return JSONResponse(content={"message": f"File '{file.filename}' uploaded successfully!"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/ask/")
async def ask_question(query: str):
    if not qa_chain_result:
        return {"error": "No PDF has been uploaded and processed yet."}
    result = qa_chain_result["qa_chain"].run(query)
    return {"answer": result}


@app.post("/explain/")
async def ask_question(query: str):
    if not qa_chain_result:
        return {"error": "No PDF has been uploaded and processed yet."}
    relevant_docs = qa_chain_result["retriever"].get_relevant_documents(query)

    for doc in relevant_docs:
        print(f"- Page Number: {doc.metadata.get('page_number', 'Unknown')}")
        print(f"  Chunk Number: {doc.metadata.get('chunk_number', 'Unknown')}")
        print(f"  Section Name: {doc.metadata.get('section_name', 'Unknown')}")
        print(f"  Content Preview: {doc.page_content[:900]}...\n") 
    return {"answer": "XAI works"}