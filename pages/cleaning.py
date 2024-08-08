import streamlit as st
import pandas as pd
import numpy as np
from docx import Document

import sys
import os

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

def clean_xlsx_file(file):
    excel_file = pd.ExcelFile(file)
    sheet_names = excel_file.sheet_names
    dfs = pd.read_excel(excel_file, sheet_name=None)
    results= []
    for sheet in sheet_names:
        result = clean_df(dfs[sheet])
        results.append(result)
    return pd.concat(results)
        

def clean_df(df):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    categories = [category for category in df.columns if "Posts in" in category]
    dfs = []

    for category in categories:

        category_df = df[category][df[category].notnull()]

        if len(category_df) == 0:
            continue

        non_category_columns = df.loc[:, ~df.columns.str.contains("Posts in")]

        # split_df = category_df.str.split(pat= r'(Mentor|Mentee)(\s+\d{4}\-\d{2}\-\d{2},?\s\d{2}:\d{2}):\s|([Mm]entor|[Mm]entee)\scommented\sat\s(\d?\d:\d{2}[AP]M\s.*\s\d?\d)', n=None, regex=True, expand=True)
        split_df = category_df.str.split(pat= r'(Mentor|Mentee)(\scommented\sat\s)?(\s+\d{4}\-\d{2}\-\d{2},?\s\d{2}:\d{2}|\d?\d:\d{2}[AP]M\s.*\s\d?\d):?\s', n=None, regex=True, expand=True)

        # this = False
        # if len(split_df.columns) == 1:
        #     split_df = category_df.str.split(pat= r'([Mm]entor|[Mm]entee)\scommented\sat\s(\d?\d:\d{2}[AP]M\s.*\s\d?\d)', n=None, regex=True, expand=True)
        #     this = True

        concat_df = pd.concat([non_category_columns, split_df], axis=1).dropna(subset=[0])

          # Select the columns that will be put into "Mentor", "Reponse Datetime", and "Response" columns.
        mentor_cols = concat_df.columns[~concat_df.columns.isin(['Mentor ID', 'Mentee ID', 'Mentor Created at', 'Relationship ID', 'Relation[Created At]', 0])][::4]
        date_cols = concat_df.columns[~concat_df.columns.isin(['Mentor ID', 'Mentee ID', 'Mentor Created at', 'Relationship ID', 'Relation[Created At]',0])][2::4]
        response_cols = concat_df.columns[~concat_df.columns.isin(['Mentor ID', 'Mentee ID', 'Mentor Created at', 'Relationship ID', 'Relation[Created At]', 0])][3::4]

        # Separate into dataframes each containing one of the new columns and the non-transformed columns
        response_df = pd.melt(concat_df, id_vars=df.columns[~df.columns.str.contains("Posts in")], value_vars=response_cols, var_name='Response_Col', value_name='Response')
        date_df = pd.melt(concat_df, id_vars=df.columns[~df.columns.str.contains("Posts in")], value_vars=date_cols, var_name='date_Col', value_name='Response Datetime')
        mentor_df = pd.melt(concat_df, id_vars=df.columns[~df.columns.str.contains("Posts in")], value_vars=mentor_cols, var_name='mentor_Col', value_name='Mentor')
        # except:
        #     response_df = pd.melt(concat_df, id_vars=['Mentor ID', 'Mentee ID'], value_vars=response_cols, var_name='Response_Col', value_name='Response')
        #     date_df = pd.melt(concat_df, id_vars=['Mentor ID', 'Mentee ID'], value_vars=date_cols, var_name='date_Col', value_name='Response Datetime')
        #     mentor_df = pd.melt(concat_df, id_vars=['Mentor ID', 'Mentee ID'], value_vars=mentor_cols, var_name='mentor_Col', value_name='Mentor')

        # Recombine all of the dataframes, drop 'date_Col', na values, and duplicates
        joined_df = pd.concat([date_df, response_df['Response'], mentor_df['Mentor']], axis=1)
        joined_df = joined_df.drop(columns=['date_Col']).dropna().drop_duplicates()

        # Add category column and set to current category
        joined_df['Category'] = category

        # Add dataframe to dfs
        dfs.append(joined_df)

        

    return pd.concat(dfs)

def clean_combined(df):
    df['Mentor Created at'] = pd.to_datetime(df['Mentor Created at'])
    df['Response Datetime'] = pd.to_datetime(df['Response Datetime'], format='mixed')
    df['Response'] = df['Response'].str.strip().dropna()
    df['Mentor'] = df['Mentor'].str.capitalize()
    df.loc[:, ['Mentor ID', 'Mentee ID']] = df[['Mentor ID', 'Mentee ID']].astype(np.int64, errors='ignore')

    df['General Category'] = df['Category'].str.replace("\n", " ").str.removeprefix("Posts in ")
    df.loc[df['General Category'].str.lower().str.contains('well.being'), 'General Category'] = 'Well Being and Self Care'
    df.loc[df['General Category'].str.lower().str.contains('studying'), 'General Category'] = 'Strategic Studying'
    df.loc[df['General Category'].str.lower().str.contains('inspiration'), 'General Category'] = 'Finding Inspiration'
    df.loc[df['General Category'].str.lower().str.contains('general'), 'General Category'] = 'General Discussion'
    df.loc[df['General Category'].str.lower().str.contains('knowing'), 'General Category'] = 'Ways of Knowing'
    df.loc[df['General Category'].str.lower().str.contains('rural.+urban'), 'General Category'] = 'From Rural to Urban'
    df.loc[df['General Category'].str.lower().str.contains('paying.+school'), 'General Category'] = 'Paying for School'
    df.loc[df['General Category'].str.lower().str.contains('dis.+mis'),'General Category'] = 'Discrimination and Misinformation'
    df.loc[df['General Category'].str.lower().str.contains('agency'), 'General Category'] = 'Agency in the World'
    df.loc[df['General Category'].str.lower().str.contains('job'), 'General Category'] = 'Getting Hired'
    df.loc[df['General Category'].str.lower().str.contains('hired'), 'General Category'] = 'Getting Hired'
    df.loc[df['General Category'].str.lower().str.contains('survey'), 'General Category'] = 'Survey'
    df.loc[df['General Category'].str.lower().str.contains('instructions'), 'General Category'] = 'Start Here!'
    df.loc[df['General Category'].str.lower().str.contains('start here'), 'General Category'] = 'Start Here!'
    df.loc[df['General Category'].str.lower().str.contains('wrapping'), 'General Category']= 'Wrapping Up'
    df.loc[df['General Category'].str.lower().str.contains('career.+tion'), 'General Category'] = 'Career Exploration'
    df.loc[df['General Category'].str.lower().str.contains('secondary'), 'General Category'] = 'Post-Secondary & Career Planning'
    df.loc[df['General Category'].str.lower().str.contains('confronting'), 'General Category'] = 'Confronting Discrimination'

    nan_relationships = df[df['Mentor ID'].isna()]['Relationship ID'].drop_duplicates()
    for id in nan_relationships:
        df.loc[df['Relationship ID'] == id, 'Mentor ID'] = df[df['Relationship ID'] == id]['Mentor ID'].median()
        df.loc[df['Relationship ID'] == id, 'Mentee ID'] = df[df['Relationship ID'] == id]['Mentee ID'].median()

    return df.reset_index().drop('index', axis=1)

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
            clean_data.append(clean_xlsx_file(file))
        elif file.name[-4:] == '.csv':
            clean_data.append(clean_df(pd.read_csv(file)))

    combined_data = clean_combined(pd.concat(clean_data))
    st.write(combined_data)