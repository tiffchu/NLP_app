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

st.title("EDA (Exploratory Data Analysis)")

if 'uploaded_file' in st.session_state:
    uploaded_file = st.session_state['uploaded_file']
    #st.write("File from Main Page:")
    #st.write(uploaded_file.head())  # Display the DataFrame head
else:
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    if isinstance(uploaded_file, pd.DataFrame):
        df = uploaded_file
    else:
        df = pd.read_csv(uploaded_file)
        st.session_state['uploaded_file'] = df  # Save the DataFrame to session state

    st.write("Uploaded CSV file:")
    st.write(df.head())
    
    columns = df.columns.tolist()

    selected_column = st.selectbox("Select column for filtering - only works for categorical columns", columns)

    if selected_column:
        categories = df[selected_column].unique()
        selected_category = st.selectbox(f"Select specific category in '{selected_column}' to filter by", categories)

        if selected_category:
            st.write(f"Filtered Data for '{selected_category}' in '{selected_column}'")
            filtered_df = df[df[selected_column] == selected_category]
            st.write(filtered_df)

        show_only_column = st.checkbox(f"Show single column (optional)")

        if show_only_column:
                selected_column1 = st.selectbox("Show specific column only)", columns)
                st.write(filtered_df[selected_column1])

                filter_text = st.checkbox(f"Filter text column (optional)")

                if filter_text:
                    text_columns = [col for col in columns if df[col].dtype == 'object']
                    selected_text_column = st.selectbox("Select RESPONSE(text) column for filtering", text_columns)

                    if selected_text_column:
                        filter_option = st.radio(
                            "Filter text based on word count",
                            ("Show all", "More than 20 words", "More than 50 words")
                        )

                        # if filter_option == "More than 20 words":
                        #     filtered_df = filtered_df[filtered_df[selected_text_column].apply(count_words) > 20]
                        # elif filter_option == "More than 50 words":
                        #     filtered_df = filtered_df[filtered_df[selected_text_column].apply(count_words) > 50]
                        if filter_option == "Show all":
                            st.write(filtered_df[filtered_df[selected_text_column].apply(count_words) > 0])
                        elif filter_option == "More than 20 words":
                            st.write(filtered_df[filtered_df[selected_text_column].apply(count_words) > 20])
                        elif filter_option == "More than 50 words":
                            st.write(filtered_df[filtered_df[selected_text_column].apply(count_words) > 50])
    st.write("")
    st.write("")    

    columns = df.columns.tolist()
    selected_columns = st.multiselect("Select numerical or categorical columns for analysis", columns)

    if selected_columns:
        st.write("Selected Columns:")
        st.write(df[selected_columns].describe())

        plot_type = st.selectbox("Select plot type", ["Histogram", "Boxplot", "Scatter Matrix", "Correlation Heatmap"])

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

    st.write("Word Count Over Time")
    date_column = st.selectbox("Select date column for time series analysis", columns)

    if date_column and date_column in df.columns:
                st.write("Calculating word count over time...")
                df[date_column] = pd.to_datetime(df[date_column])
                text_column = st.selectbox("Select text column for word count", selected_columns)

                if text_column and text_column in df.columns:
                    df['Word Count'] = df[text_column].apply(count_words)
                    df['DateGroup'] = df[date_column].dt.to_period('Q')
                    word_count_over_time = df.groupby('DateGroup')['Word Count'].sum()
                    fig, ax = plt.subplots()
                    ax.plot(word_count_over_time.index.to_timestamp(), word_count_over_time.values, marker='o', linestyle='-')
                    ax.set_xlabel("Date")
                    ax.set_ylabel('Total Word Count')
                    ax.set_title('Total Word Count Over Time (3-month periods)')
                    plt.xticks(rotation=70)
                    st.pyplot(fig)
    
    date_column = st.selectbox("Select date column for time series analysis (if any)", ["None"] + columns)

    if date_column != "None":
        df[date_column] = pd.to_datetime(df[date_column])
        time_series_column = st.selectbox("Select column for time series analysis", [col for col in columns if col != date_column])

        if time_series_column:
            st.write(f"Time Series Analysis for {time_series_column}")
            fig, ax = plt.subplots()
            df.set_index(date_column)[time_series_column].plot(ax=ax)
            st.pyplot(fig)
