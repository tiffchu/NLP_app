import streamlit as st
import pandas as pd
from docx import Document

import sys
import os

# # Get the parent directory
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# # Add the parent directory to sys.path
# sys.path.append(parent_dir)

# # Now you can import modules from the parent directory
# from utils.file_cleaners import clean_docx_file

def update_raw_data(raw_data):
    st.session_state.raw_data = raw_data

def clean_docx_file(file):
    doc = Document(file)
    relationship_id = int(file.name.lower().lstrip("relationship ").rstrip(".docx"))
    text = []
    categories = {}
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            if run.font.highlight_color:
                categories[run.text] = ''.join(text)
                text = []
            else:
                text.append(run.text)

    input = {'Category': categories.keys(), 'Responses': categories.values()}
    df = pd.DataFrame.from_dict(input)
    df = df[df['Category'].astype('string') != '']

    category_column = df[['Category']]

    split_df = df['Responses'].str.split(pat= r'([Mm]entor|[Mm]entee)\scommented\sat\s(\d?\d:\d\d[AP]M\s\w+\s\d?\d)', n=None, regex=True, expand=True)

    concat_df = pd.concat([category_column, split_df], axis=1)

    # Select the columns that will be put into "Mentor", "Reponse Datetime", and "Response" columns.
    response_cols = split_df.columns[3::3]
    mentor_cols = split_df.columns[1::3]
    date_cols = split_df.columns[2::3]

    response_df = pd.melt(concat_df, id_vars=['Category'], value_vars=response_cols, var_name='Response_Col', value_name='Response')
    date_df = pd.melt(concat_df, id_vars=['Category'], value_vars=date_cols, var_name='date_Col', value_name='Response Datetime')
    mentor_df = pd.melt(concat_df, id_vars=['Category'], value_vars=mentor_cols, var_name='mentor_Col', value_name='Mentor') 


    joined_df = pd.concat([date_df, response_df['Response'], mentor_df['Mentor']], axis=1)
    joined_df = joined_df.drop(columns=['date_Col']).dropna().drop_duplicates() 

    joined_df['Relationship ID'] = relationship_id

    return joined_df

st.title("NLP Preprocessing and Topic Modeling App")
st.write("""
Upload a CSV file, select a text column for NLP preprocessing,
and perform topic modeling to visualize the topics using BERTopic.
""")

files = st.file_uploader("Choose files containing data", type=['csv', 'xlsx', 'docs', 'docx'], accept_multiple_files=True)

if files:
    select_data_button = st.button("Clean and Combine Data", key="select_data_button", on_click=update_raw_data, kwargs={'raw_data': files})

if 'raw_data' in st.session_state:
    files = st.session_state.raw_data
    clean_data = []
    for file in files:
        if file.name[-5:] == '.docx':
            clean_data.append(clean_docx_file(file))
        elif file.name[-5:] == '.xlsx':
            st.write(file.name)
        elif file.name[-4:] == '.csv':
            st.write(file.name)

    for data in clean_data:
        st.write(data)