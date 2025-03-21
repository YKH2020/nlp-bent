import streamlit as st
from llm_inference import run_inference
import gdown
import zipfile
import os

def run_model(user_prompt):
    '''
    Perform Inference
    '''
    if not os.path.isdir("finetuned-philosophers-llama3.2-1b"):
        url = "https://drive.google.com/uc?export=download&id=1yO7usn03IBb1VTPe3qW2BIXkH5wxHSMA"
        gdown.download(url, 'model_weights.zip', quiet=False)
        with zipfile.ZipFile('model_weights.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        os.rename("finetuned-philosophers-llama3.2-1b/model-002.safetensors", "finetuned-philosophers-llama3.2-1b/model.safetensors")
    return run_inference("finetuned-philosophers-llama3.2-1b", user_prompt)

st.title("Philosophy LLM")
st.write("This website allows you to prompt an LLM fine tuned on a variety of philosophical schools of thought.")

st.header("Getting started")
st.write("To get started on using the models below, start by asking the model anything related to philosophy")

user_prompt = st.text_area("Enter your prompt:", "")

if st.button("Submit"):
    with st.spinner(text="Processing", show_time=True):
        output = run_model(user_prompt)

    st.write("Output")
    st.write(output)
