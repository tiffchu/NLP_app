import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Natural Language Processing App",
    page_icon="🤖",
    initial_sidebar_state='auto',
    
)

st.title("This is the Home Page.")
st.write("Welcome to the Natural Language Processing App!")
st.write("Use the sidebar to navigate to different sections of the app.")


st.write("Upload a CSV file")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['uploaded_file'] = df  #save the uploaded file to session state
    st.write("File uploaded successfully! You can now navigate to the NLP tools on the sidebar to work with this file.")
    st.write("Uploaded CSV file:")
    st.write(df.head())
