import json
import os
import streamlit as st
from streamlit_chat import message
import boto3

smr_client = boto3.client("sagemaker-runtime")
os.environ["endpoint_name"] = "enter endpoint name here"
os.environ["llama_ic_name"] = "enter llama IC name here"
os.environ["bart_ic_name"] = "enter bart IC name here"

# Sidebar to summarize and clear conversation
st.sidebar.title("Sidebar")
clear_button = st.sidebar.button("Clear Conversation", key="clear")
summarize_button = st.sidebar.button("Summarize Conversation", key="summarize")

# containers for input and claude response
container = st.container()
response_container = st.container()


def invoke_llama(payload: dict, endpoint_name: str = os.environ.get("endpoint_name"), content_type: str = "application/json",
                 ic_name: str = os.environ.get("llama_ic_name")):
    response = smr_client.invoke_endpoint(
        EndpointName=endpoint_name,
        InferenceComponentName=ic_name, #specify IC name
        ContentType=content_type,
        Body=json.dumps(payload),
    )
    result = json.loads(response['Body'].read().decode())
    return result

# session state variables store user and model inputs
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# for summarization
if summarize_button:
    st.header("Summary")
    st.write("Generating summary....")
    chat_history = st.session_state['chat_history']
    text = ''''''
    for resp in chat_history:
        if resp['role'] == "user":
            text += f"Ram: {resp['content']}\n"
        elif resp['role'] == "assistant":
            text += f"AI: {resp['content']}\n"
    summary_payload = {"inputs": text}
    summary_response = smr_client.invoke_endpoint(
        EndpointName=os.environ.get("endpoint_name"),
        InferenceComponentName=os.environ.get("bart_ic_name"), #specify IC name
        ContentType="application/json",
        Body=json.dumps(summary_payload),
    )
    summary_result = json.loads(summary_response['Body'].read().decode())
    summary = summary_result[0]['summary_text']
    st.write(summary)
    
# reset everything upon clear
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['chat_history'] = []

# reference: https://github.com/marshmellow77/falcon-document-chatbot/blob/main/chatbot.py
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')
        #when user inputs, execute chain
        if submit_button and user_input:
            st.session_state['past'].append(user_input)
            model_input = {"role": "user", "content": user_input}
            st.session_state['chat_history'].append(model_input)
            payload = {"chat": st.session_state['chat_history'], "parameters": {"max_tokens":400, "do_sample": True,
                                                                                "maxOutputTokens": 2000}}
            response = smr_client.invoke_endpoint(
                EndpointName=os.environ.get("endpoint_name"),
                InferenceComponentName=os.environ.get("llama_ic_name"), #specify IC name
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            full_output = json.loads(response['Body'].read().decode())
            print(full_output)
            display_output = full_output['content']
            print(display_output)
            st.session_state['chat_history'].append(full_output)
            st.session_state['generated'].append(display_output)


if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))