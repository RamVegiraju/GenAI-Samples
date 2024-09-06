import boto3
import json
import os
from pypdf import PdfReader
import streamlit as st

client = boto3.client('bedrock')
runtime = boto3.client('bedrock-runtime')

def model_inference(pdf_text: str, model_id: str = 'anthropic.claude-3-5-sonnet-20240620-v1:0', accept: str = "application/json", contentType = "application/json",
                    anthropic_version: str = "bedrock-2023-05-31", max_tokens: int = 2000):
    summarize_prompt = f"""Summarize the following text: {pdf_text} in two sentences max."""
    text_content = [{'type':'text','text': summarize_prompt}]
    text_payload = {"messages":[{"role":"user","content":text_content}], "anthropic_version": anthropic_version, "max_tokens": max_tokens}
    response = runtime.invoke_model(
        body=json.dumps(text_payload), modelId=model_id, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())
    summary = response_body['content'][0]['text']
    return summary


st.title("Document Summarizer Using Claude Sonnet 3.5 via Amazon Bedrock")
# pdf uploading for streamlit: https://discuss.streamlit.io/t/how-to-upload-a-pdf-file-in-streamlit/2428/2
uploaded_file = st.file_uploader('Upload your document (must be PDF)', type="pdf")
if uploaded_file is not None:
    try:
        # pdf parsing code borrowed from: https://github.com/anthropics/anthropic-cookbook/blob/main/misc/pdf_upload_summarization.ipynb
        reader = PdfReader(uploaded_file)
        number_of_pages = len(reader.pages)
        text = ''.join(page.extract_text() for page in reader.pages)
    except Exception as e:
        raise Exception(f"There was an error parsing the PDF: {e}")
    summary = model_inference(text)
    st.title("Article Summary: ")
    st.write(summary)