# Import necessary libraries
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Function to summarize text
def summarize_text(text, max_length=512, summary_length=50):
    # Split text into smaller chunks
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    input_ids = inputs['input_ids']

    # Generate summary for each chunk
    summary_ids = model.generate(
        input_ids,
        max_new_tokens=summary_length,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    
    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("skeskinen/llama-lite-134m")
model = AutoModelForCausalLM.from_pretrained("skeskinen/llama-lite-134m")

# Streamlit app
st.title("Text Summarization using LLaMA Model")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Dataframe:")
    st.write(df)
    
    # Select text column
    columns = df.columns.tolist()
    text_column = st.selectbox("Select the text column", columns)
    
    if text_column:
        # Combine all text in the selected column
        combined_text = " ".join(df[text_column].dropna().astype(str).tolist())
        
        # Perform summarization on chunks
        chunk_size = 500  # Define chunk size
        combined_summary = ""
        
        for i in range(0, len(combined_text), chunk_size):
            chunk = combined_text[i:i+chunk_size]
            summary = summarize_text(chunk)
            combined_summary += summary + " "
        
        st.write("Summary:")
        st.write(combined_summary.strip())
