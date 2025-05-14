#Ollama 모델 로드 및 테스트
from langchain_community.chat_models import ChatOllama
import gradio as gr
import numpy as np

model = ChatOllama(model="deepseek-r1:latest", temperature=0)

def echo(message, history):
    response = model.invoke(message)
    return response.content

demo = gr.ChatInterface(fn=echo, title="훌륭한 친구들")
demo.launch(share=True)