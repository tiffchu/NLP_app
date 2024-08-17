import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Natural Language Processing App",
    page_icon="🤖",
    initial_sidebar_state='auto',
    
)

#st.title("This is the Home Page.")
st.title("Welcome to the Natural Language Processing App!")
st.write("Use the sidebar to navigate to different sections of the app.")


st.write("Upload a CSV file")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['uploaded_file'] = df  #save the uploaded file to session state
    st.write("File uploaded successfully! You can now navigate to the NLP tools on the sidebar to work with this file.")
    st.write("Uploaded CSV file:")
    st.write(df.head())


st.write('Find out more about each tool in the column bar on the left:')

st.write("*Cleaning:*")
st.markdown("**Upload dataset(s) in .word, .xlsx, or .csv format, and it'll separate responses that are in the same cell into different rows**")

st.write("*Filtering and Analysis:*")
st.markdown("**Filter for text of a specific category, or a specific word length. See visualizations of the words or the columns**")

st.write("*Abstractive Summarization:*")
st.markdown("**Copy and paste paragraphs of text into our website, and it'll output a summarized version of the text**")

st.write("*Mood Detection:*")
st.markdown("**Mood detection provides us with the magnitudes of each of the inputted emotions from a given dataset**")

st.write("*Topic Modelling BERT and LDA:*")
st.markdown("**Topic Modeling, which is the process of extracting common themes from text, by picking groups of similar words that often appear together in texts. BERT and LDA are two different algorithms used for topic modelling, we prefer the former**")

