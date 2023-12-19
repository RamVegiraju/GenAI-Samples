import json
import streamlit as st
from streamlit_chat import message
import boto3
import langchain
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

client = boto3.client('bedrock')
runtime = boto3.client('bedrock-runtime')

# Sidebar to clear conversation
st.sidebar.title("Sidebar")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# containers for input and claude response
container = st.container()
response_container = st.container()

# we only want this executed once, global resource across all reruns and users
@st.cache_resource
def load_chain():
    modelId = "anthropic.claude-v2"
    # setup prompt for claude, this will differ depending on model
    claude_prompt = PromptTemplate.from_template("""

    Human: The following is a friendly conversation between a human and an AI.
    The AI is talkative and provides lots of specific details from its context. If the AI does not know
    the answer to a question, it truthfully says it does not know.

    Current conversation:
    <conversation_history>
    {history}
    </conversation_history>

    Here is the human's next reply:
    <human_reply>
    {input}
    </human_reply>

    Assistant:
    """)
    llm = Bedrock(model_id=modelId, model_kwargs={"max_tokens_to_sample": 1000})
    chain = ConversationChain(llm=llm, verbose=False, memory=ConversationBufferMemory(), prompt=claude_prompt)
    return chain

# Initilialize chain 
chain = load_chain()

# session state variables store user and model inputs
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    chain.memory.clear()

# reference: https://github.com/marshmellow77/falcon-document-chatbot/blob/main/chatbot.py
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')
        #when user inputs, execute chain
        if submit_button and user_input:
            model_output = chain.predict(input = user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(model_output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))