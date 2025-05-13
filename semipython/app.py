# app.py
import gradio as gr
import os
import tempfile
from typing import List, Dict, Any

from ollama_client import MultiModelSystem
from vector_store import VectorStore
from multi_agent import MultiAgentSystem

# Initialize components
models = MultiModelSystem()
vector_store = VectorStore()
agent_system = MultiAgentSystem(models, vector_store)

# Chat history for the session
chat_history = []

def process_message(message: str, history: List[List[str]]) -> str:
    """Process user message and update chat history"""
    global chat_history
    
    # Convert Gradio history format to our format
    if not chat_history:
        for human, ai in history:
            chat_history.append({"role": "user", "content": human})
            if ai:  # Some might be None if this is a new message
                chat_history.append({"role": "assistant", "content": ai})
    
    # Add the new message
    chat_history.append({"role": "user", "content": message})
    
    # Process query through the agent system
    response = agent_system.process_query(message, chat_history)
    
    # Add the response to the chat history
    chat_history.append({"role": "assistant", "content": response})
    
    return response

def upload_pdf(file):
    """Handle PDF upload and processing"""
    if file is None:
        return "No file uploaded."
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_path = temp_file.name
            temp_file.write(file)
        
        # Process PDF with vector store
        doc_id = vector_store.add_pdf(temp_path, {"source": file.name})
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return f"PDF '{file.name}' successfully processed and added to the knowledge base with ID: {doc_id}"
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Multi-Agent RAG System") as demo:
    gr.Markdown("# Multi-Agent RAG System with Ollama Models")
    
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Message")
        clear = gr.Button("Clear")
        
        msg.submit(process_message, [msg, chatbot], [chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
    
    with gr.Tab("Upload PDF"):
        pdf_input = gr.File(label="Upload PDF Document", file_types=[".pdf"])
        upload_button = gr.Button("Process PDF")
        result_text = gr.Textbox(label="Result")
        
        upload_button.click(upload_pdf, inputs=[pdf_input], outputs=[result_text])
    
    gr.Markdown("""
    ## 시스템 사용 방법:
    
    1. **PDF 업로드**: 'Upload PDF' 탭에서 문서를 지식 베이스에 추가하세요.
    2. **채팅**: 채팅 탭에서 질문하면 시스템이 적절한 모델을 사용하여 답변합니다.
    3. **모델**: 시스템은 코드 관련 질문에 deepseek-r1, 복잡한 추론에 llama3.2, 일반 질문에 gemma를 사용합니다.
    """)

if __name__ == "__main__":
    demo.launch()
