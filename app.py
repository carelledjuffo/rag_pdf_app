import gradio as gr
import requests

fastapi_url = "http://localhost:8000"
def upload_pdf_to_fastapi(file):
    # The backend URL of FastAPI
    
    # Send the file to FastAPI via a POST request
    with open(file.name, "rb") as f:
        response = requests.post(f"{fastapi_url}/upload-pdf/", files={"file": f})
        
    return response.json().get("message", "Failed to upload")

def ask_question_from_pdf(question):
    res = requests.post(f"{fastapi_url}/ask/", params={"query": question})
    return res.json().get("answer", "Error or no PDF uploaded yet.")

def explain_answer_from_pdf(question):
    res = requests.post(f"{fastapi_url}/explain/", params={"query": question})
    return res.json().get("answer", "Error or no PDF uploaded yet.")

with gr.Blocks() as demo:
    gr.Markdown("# PDF Q&A (RAG-powered)")

    with gr.Row():
        file_input = gr.File(type="filepath", label="Upload PDF")
        upload_button = gr.Button("Upload")

    upload_output = gr.Textbox(label="Upload Status")
    upload_button.click(fn=upload_pdf_to_fastapi, inputs=file_input, outputs=upload_output)
    global question_input 
    with gr.Row():
        question_input = gr.Textbox(label="Ask a Question")
        send_question_button = gr.Button("Ask")
    answer_output = gr.Textbox(label="Answer")
    send_question_button.click(fn=ask_question_from_pdf, inputs=question_input, outputs=answer_output)

    with gr.Row():
        explain_question_button = gr.Button("Explain")
        explain_output = gr.Textbox(label="XAI")
    explain_question_button.click(fn=explain_answer_from_pdf, inputs=question_input, outputs=explain_output)

demo.launch()