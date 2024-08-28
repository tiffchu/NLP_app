import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="User Matching Recommender System",
    page_icon="😊",
    initial_sidebar_state='auto',
)

st.title("User Matching Recommender System")
st.write("""
Upload a CSV file with a user profile in each row
""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['uploaded_file'] = df
else:
    df = st.session_state['uploaded_file']

if df is not None:
    st.write("Uploaded CSV file:")
    st.write(df.head())