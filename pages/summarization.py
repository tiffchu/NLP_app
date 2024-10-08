import streamlit as st
import torch
from transformers import PegasusForConditionalGeneration, AutoTokenizer
from transformers import PegasusTokenizer

st.set_page_config(
    page_title="Abstractive Summarization App",
    page_icon="✍️",
    initial_sidebar_state='auto',
)

model_name = 'google/pegasus-multi_news'

st.title("Abstractive Summarization with Pegasus")
st.write("Enter a paragraph of text to get its abstractive summary.")
st.write("fine-tuned on:", model_name)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

def summarize_text(text):
    batch = tokenizer(text, truncation=True, padding='longest', return_tensors="pt").to(device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text[0]

text = st.text_area("Input Paragraph", height=200)

if st.button("Summarize"):
    if text.strip():
        summary = summarize_text(text)
        st.write("**Summary:**")
        st.write(summary)
    else:
        st.write("Please enter some text in the input area.")
