from openai import OpenAI
import streamlit as st

client = OpenAI(api_key="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", base_url="https://openrouter.ai/api/v1")

while True:
    message = st.text_area("Write Your Message here")
    
    

