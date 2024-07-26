import streamlit as st
import pandas as pd
import torch

from transformers import (
    pipeline,
    DistilBertForQuestionAnswering,
    BertForQuestionAnswering,
    RobertaForQuestionAnswering,
    AlbertForQuestionAnswering,
    XLNetForQuestionAnswering,
    ElectraForQuestionAnswering,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
)

MODEL_OPTIONS = {
    "distilbert-base-cased-distilled-squad": "distilbert-base-cased-distilled-squad",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "bert-large-uncased-whole-word-masking-finetuned-squad",
    # "roberta-base-squad2": "roberta-base-squad2",
    # "albert-xlarge-v2-squad-v2": "albert-xlarge-v2-squad-v2",
    # "xlnet-large-cased": "xlnet-large-cased",
    # "electra-large-discriminator-squad2": "electra-large-discriminator-squad2",
    # "t5-base-qa-summary": "t5-base-qa-summary",
}

st.title("Question Answering with NLP Models")

if 'uploaded_file' in st.session_state:
    uploaded_file = st.session_state['uploaded_file']
else:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    if isinstance(uploaded_file, pd.DataFrame):
        df = uploaded_file
    else:
        df = pd.read_csv(uploaded_file)
        st.session_state['uploaded_file'] = df 

    st.write("Uploaded CSV file:")
    st.write(df.head())

    columns = df.columns.tolist()

    text_column = st.selectbox("Select the text column for context data", df.columns)

    model_name = st.selectbox("Choose a pre-trained model", list(MODEL_OPTIONS.keys()))

    model_path = MODEL_OPTIONS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)

    def concatenate_context(column_data):
        concatenated_text = " ".join(column_data.astype(str).tolist())
        return concatenated_text

    context = concatenate_context(df[text_column])
    question = st.text_area("Enter your question here")

    if st.button("Answer"):
        if text_column and len(df) > 0:
            inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1

            answer = tokenizer.decode(input_ids[0, answer_start:answer_end], skip_special_tokens=True)
            st.write(f"Answer: {answer}")
        else:
            st.write("Please upload a CSV file and select the text column.")

else:
    st.write("Please upload a CSV file to get started.")
