import streamlit as st

st.set_page_config(
    page_title="Natural Language Processing App",
    page_icon="🤖",
    initial_sidebar_state='auto',
    
)

def home_page():
    st.title("This is the Home Page.")
    st.write("Welcome to the Natural Language Processing App!")
    st.write("Use the sidebar to navigate to different sections of the app.")

home_page()
