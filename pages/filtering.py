import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def count_words(text):
    # Function to count words in a text
    if isinstance(text, str):
        return len(text.split())
    return 0


def calculate_relationship_duration_and_responses(df):
    df['Response Datetime'] = pd.to_datetime(df['Response Datetime'])
    
    # Cumulative count of responses
    df = df.sort_values(by=['Mentor ID', 'Mentee ID', 'Relationship ID', 'Response Datetime'])
    df['Current Number of Responses'] = df.groupby(['Mentor ID', 'Mentee ID', 'Relationship ID']).cumcount() + 1
    
    # Calculate word count for each response
    df['Word Count'] = df['Response'].apply(count_words)
    
    # Calculate cumulative word count for each relationship
    df['Cumulative Word Count'] = df.groupby(['Mentor ID', 'Mentee ID', 'Relationship ID'])['Word Count'].cumsum()
    
    grouped_df = df.groupby(['Mentor ID', 'Mentee ID', "Relationship ID"]).agg(
        First_Response=('Response Datetime', 'min'),
        Last_Response=('Response Datetime', 'max'),
        Total_Responses=('Response', 'count')
    ).reset_index()
    
    grouped_df['Relationship Duration'] = grouped_df['Last_Response'] - grouped_df['First_Response']
    
    # Merge the grouped_df with the original df to include the new columns
    df = pd.merge(df, grouped_df[['Mentor ID', 'Mentee ID', 'Relationship ID', 'Relationship Duration', 'Total_Responses']],
                  on=['Mentor ID', 'Mentee ID', 'Relationship ID'], how='left')
    
    return df


st.title("Dataset Filtering and Exploratory Data Analysis")
st.write("Works only with cleaned datasets")

if 'uploaded_file' in st.session_state:
    uploaded_file = st.session_state['uploaded_file']
    #st.write("File from Main Page:")
    #st.write(uploaded_file.head()) 
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
    
    result_df = calculate_relationship_duration_and_responses(df)
    df = pd.merge(df, result_df, on=['Mentor ID', 'Mentee ID', 'Relationship ID'], how='left')

    df = df.drop(columns=['Response_x','Mentor Created at_x','Mentor Created at_y','Mentor_x', 'Category_x','Response Datetime_x'])
    
    st.write("Adding Relationship Duration and Number of Responses in the relationship so far:")

    st.write(df)

    columns = df.columns.tolist()

    selected_column = st.selectbox("Select column for filtering - only works for categorical columns", columns)

    if selected_column:
        categories = df[selected_column].unique()
        selected_category = st.selectbox(f"Select specific category in '{selected_column}' to filter by", categories)

        if selected_category:
            st.write(f"Filtered Data for '{selected_category}' in '{selected_column}'")
            filtered_df = df[df[selected_column] == selected_category]
            st.write(filtered_df)

        show_only_column0 = st.checkbox(f"Show single column (optional)", key="show_only_column0")

        if show_only_column0:
                selected_column1 = st.selectbox("Show specific column only)", columns, key="selected_column1")
                st.write(filtered_df[selected_column1])

    text_columns = [col for col in columns if df[col].dtype == 'object']

    selected_text_column = st.selectbox("Select text('response') column for filtering", text_columns)

    filter_option = st.radio(
        "Filter text based on word count",
        ("Show all", "More than [a custom number] of words")
    )

    if filter_option == "Show all":
        filtered_data = filtered_df[filtered_df[selected_text_column].apply(count_words) > 0]
    elif filter_option == "More than [a custom number] of words":
        custom_word_count = st.number_input("Enter the minimum number of words:", min_value=1, value=20)
        filtered_data = filtered_df[filtered_df[selected_text_column].apply(count_words) > custom_word_count]

    st.write(filtered_data)

    show_only_column2 = st.checkbox("Show single column (optional)", key="show_only_column2")

    if show_only_column2:
        selected_column2 = st.selectbox("Select column to display", columns, key="selected_column2")
        st.write(filtered_data[selected_column2])

    st.write("")
        

    columns = df.columns.tolist()

st.header("Visualize Categorical or numerical columns")

def plot_word_count_over_time(df, selected_relationship_id):
    st.write(f"Word Count Progression Over Time for Relationship ID {selected_relationship_id}")
    plt.figure(figsize=(10, 6))
    
    subset = df[df['Relationship ID'] == selected_relationship_id]
    plt.plot(subset['Response Datetime_y'], subset['Cumulative Word Count'], label=f'Relationship {selected_relationship_id}')
    
    plt.title(f'Word Count Over Time for Relationship {selected_relationship_id}')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Word Count')
    plt.legend(title='Relationship ID')
    plt.grid(True)
    st.pyplot(plt)

def plot_responses_over_time(df, selected_relationship_id):
    st.write(f"Number of Responses Over Time for Relationship ID {selected_relationship_id}")
    plt.figure(figsize=(10, 6))
    
    subset = df[df['Relationship ID'] == selected_relationship_id]
    plt.plot(subset['Response Datetime_y'], subset['Current Number of Responses'], label=f'Relationship {selected_relationship_id}')
    
    plt.title(f'Number of Responses Over Time for Relationship {selected_relationship_id}')
    plt.xlabel('Time')
    plt.ylabel('Number of Responses')
    plt.legend(title='Relationship ID')
    plt.grid(True)
    st.pyplot(plt)



relationship_ids = df['Relationship ID'].unique()
selected_relationship_id = st.selectbox("Select Relationship ID to visualize", relationship_ids)
    
plot_word_count_over_time(df, selected_relationship_id)
    
plot_responses_over_time(df, selected_relationship_id)



st.header("Visualize More Categorical or numerical columns")
selected_columns = st.multiselect("Select:", columns)

if selected_columns:
        st.write("Select numerical or categorical columns for analysis")
        st.write(df[selected_columns].describe())

        plot_type = st.selectbox("Select plot type", ["Bar", "Histogram", "Boxplot", "Scatter Matrix", "Correlation Heatmap"])

        if plot_type == "Bar":
            for column in selected_columns:
                st.write(f"Bar Plot for {column}")
                fig, ax = plt.subplots()
                
                # Check if the column is categorical or numerical
                if df[column].dtype == 'object' or df[column].nunique() < 20:
                    df[column].value_counts().plot(kind='bar', ax=ax)
                else:
                    df[column].plot(kind='bar', ax=ax)
                    
                ax.set_title(f'Bar Plot for {column}')
                ax.set_xlabel(column)
                ax.set_ylabel('Count')
                st.pyplot(fig)

        if plot_type == "Histogram":
            for column in selected_columns:
                st.write(f"Histogram for {column}")
                fig, ax = plt.subplots()
                df[column].hist(ax=ax, bins=30)
                st.pyplot(fig)

        elif plot_type == "Boxplot":
            for column in selected_columns:
                st.write(f"Boxplot for {column}")
                fig, ax = plt.subplots()
                sns.boxplot(x=df[column], ax=ax)
                st.pyplot(fig)

        elif plot_type == "Scatter Matrix":
            st.write("Scatter Matrix")
            fig = sns.pairplot(df[selected_columns])
            st.pyplot(fig)

        elif plot_type == "Correlation Heatmap":
            st.write("Correlation Heatmap")
            fig, ax = plt.subplots()
            sns.heatmap(df[selected_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        columns = df.columns.tolist()
        selected_columns = st.multiselect("Select text column for analysis", columns)


    # st.header("Word Count Over Time")
    # date_column = st.selectbox("Select date column for time series analysis", columns)

    # if date_column and date_column in df.columns:
    #             st.write("Calculating word count over time...")
    #             df[date_column] = pd.to_datetime(df[date_column])
    #             text_column = st.selectbox("Select text column for word count", selected_columns)

    #             if text_column and text_column in df.columns:
    #                 df['Word Count'] = df[text_column].apply(count_words)
    #                 df['DateGroup'] = df[date_column].dt.to_period('Q')
    #                 word_count_over_time = df.groupby('DateGroup')['Word Count'].sum()
    #                 fig, ax = plt.subplots()
    #                 ax.plot(word_count_over_time.index.to_timestamp(), word_count_over_time.values, marker='o', linestyle='-')
    #                 ax.set_xlabel("Date")
    #                 ax.set_ylabel('Total Word Count')
    #                 ax.set_title('Total Word Count Over Time (3-month periods)')
    #                 plt.xticks(rotation=70)
    #                 st.pyplot(fig)
    
    # date_column = st.selectbox("Select date column for time series analysis (if any)", ["None"] + columns)

    # if date_column != "None":
    #     df[date_column] = pd.to_datetime(df[date_column])
    #     time_series_column = st.selectbox("Select column for time series analysis", [col for col in columns if col != date_column])

    #     if time_series_column:
    #         st.write(f"Time Series Analysis for {time_series_column}")
    #         fig, ax = plt.subplots()
    #         df.set_index(date_column)[time_series_column].plot(ax=ax)
    #         st.pyplot(fig)
