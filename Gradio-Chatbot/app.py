import gradio as gr
import os
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from utils import llm, claude_prompt

chain = ConversationChain(
    llm=llm, verbose=False, memory=ConversationBufferMemory(), prompt=claude_prompt
)

def model_inference(message, history):
    resp = chain.predict(input = message)
    return resp

demo = gr.ChatInterface(model_inference)

if __name__ == "__main__":
    demo.launch()
