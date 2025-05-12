#Ui/Ux 편집 페이지
from langchain_community.chat_models import ChatOllama
import gradio as gr

model = ChatOllama(model="llama3.2:latest", temperature=0)

def echo(message, history):
    response = model.invoke(message)
    return response.content

demo = gr.ChatInterface(fn=echo, title="성결대학교")
demo.launch()