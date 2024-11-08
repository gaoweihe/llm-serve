import streamlit as st
import requests

# Title for the Streamlit app
st.title("LLM Text Generation with FastAPI Backend")

# Input field for the prompt
prompt = st.text_input("Enter your prompt:")

# Function to make request to FastAPI backend
def get_generated_text(prompt):
    response = requests.post(
        "http://localhost:8000/generate",  # FastAPI endpoint URL
        json={"prompt": prompt}            # Sending JSON payload with the prompt
    )
    if response.status_code == 200:
        return response.json().get("generated_text", "No response received.")
    else:
        return f"Error: {response.status_code}"

# Generate button to trigger API call
if st.button("Generate"):
    if prompt:
        output = get_generated_text(prompt)
        st.write(output)
    else:
        st.write("Please enter a prompt.")
