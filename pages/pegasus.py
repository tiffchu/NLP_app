import streamlit as st
import pandas as pd
from transformers import PegasusForConditionalGeneration, AutoTokenizer
import torch

st.set_page_config(
    page_title="Abstractive Summarization App",
    page_icon="🥳",
    initial_sidebar_state='auto',
)

model_name = 'google/pegasus-xsum'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

def summarize_text(text):
    batch = tokenizer(text, truncation=True, padding='longest', return_tensors="pt").to(device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text[0]

st.title("Abstractive Summarization with Pegasus")
st.write("Upload a CSV file, select a column, and summarize either a specific row or the entire column.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded CSV file:")
    st.write(df.head())

    column = st.selectbox("Select the column to summarize", df.columns)
        
    option = st.radio("Summarize:", ("Specific row", "Entire column"))

    if option == "Specific row":
        row_number = st.number_input("Select row number", min_value=0, max_value=len(df) - 1, step=1)
        text = df[column].iloc[row_number]

        st.write("Selected Text:")
        st.write(text)

        if st.button("Summarize Row"):
            if text:
                summary = summarize_text(text)
                st.write("**Summary:**")
                st.write(summary)
            else:
                st.write("The selected row is empty.")
    
    elif option == "Entire column":
        if st.button("Summarize Entire Column"):
            summaries = []
            for text in df[column].dropna():
                summaries.append(summarize_text(text))
            st.write("**Summaries:**")
            for i, summary in enumerate(summaries):
                st.write(f"**Summary {i + 1}:** {summary}")
