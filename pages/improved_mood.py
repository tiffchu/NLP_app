import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    initial_sidebar_state='auto',
)

st.title("Emotion Detection with BART")
st.write("""
Upload a CSV file, select a text column for NLP preprocessing,
and perform emotion detection.
""")

if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state['uploaded_file'] = df
else:
    df = st.session_state['uploaded_file']

if df is not None:
    st.write("Uploaded CSV file:")
    st.write(df.head())

    text_column = st.selectbox("Select the text column for emotion detection", df.columns)

    model_name = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    mood_categories = [
        "neutral", "happiness", "excitement", "gratitude", "contempt", "sadness", "nervousness",
        "stress", "boredom", "anger", "fear", "surprise", "disgust", "calmness",
        "curiosity", "envy", "guilt", "shame", "pride", "love", "loneliness",
        "hope", "despair", "relief", "anticipation", "trust", "confusion",
        "frustration", "embarrassment", "amusement", "jealousy", "Add emotion"
    ]

    selection = st.multiselect('Select a range of emotions to predict (pick 3 emotions)', options=mood_categories)

    if "Add emotion" in selection:
        selection.remove("Add emotion")
        custom_emotion = st.text_input("Enter your custom emotion")
        if custom_emotion:
            selection.append(custom_emotion)

    def classify_text(texts, labels):
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()
        scores = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
        return scores

    def aggregate_text_by_year(dataframe, text_col, date_col):
        dataframe[date_col] = pd.to_datetime(dataframe[date_col])
        dataframe['Year'] = dataframe[date_col].dt.year
        dataframe[text_col] = dataframe[text_col].astype(str).fillna('')  #
        return dataframe.groupby('Year')[text_col].apply(' '.join).reset_index()
    
view_categories = ["View each mood individually (get the moods of all users in a table view)", "View combined/average moods of all users together (select 3 moods)", "Measure how the mood changes over time for a single relationship"]
view_format = st.radio('Select viewing format', options=view_categories)

if view_format == "Measure how the mood changes over time for a single relationship":
    if 'relationship_ids' not in st.session_state:
        st.session_state.relationship_ids = df['Relationship ID'].unique().tolist()
    
    if 'selected_relationship_id' not in st.session_state:
        st.session_state.selected_relationship_id = st.session_state.relationship_ids[0] if st.session_state.relationship_ids else None

    st.session_state.selected_relationship_id = st.selectbox(
        "Select or enter a Relationship ID",
        st.session_state.relationship_ids,
        index=st.session_state.relationship_ids.index(st.session_state.selected_relationship_id) if st.session_state.selected_relationship_id in st.session_state.relationship_ids else 0
    )

    if 'selected_role' not in st.session_state:
        st.session_state.selected_role = "Mentee"

    st.session_state.selected_role = st.radio('Select viewing format', options=["Mentee", "Mentor"], index=0 if st.session_state.selected_role == "Mentee" else 1)

    if st.button("Measure Mood Changes Over Time", key='1'):
        relationship_df = df[df['Relationship ID'] == st.session_state.selected_relationship_id]

        if st.session_state.selected_role == "Mentee":
            relationship_df = relationship_df[relationship_df['Mentor'] == 'Mentee']
        elif st.session_state.selected_role == "Mentor":
            relationship_df = relationship_df[relationship_df['Mentor'] == 'Mentor']

        if not relationship_df.empty:
            relationship_df['Response Datetime'] = pd.to_datetime(relationship_df['Response Datetime'])
            relationship_df = relationship_df.sort_values(by='Response Datetime')

            texts = relationship_df[text_column].astype(str).tolist()
            timestamps = relationship_df['Response Datetime'].tolist()

            scores = []
            for text in texts:
                text_scores = classify_text([text], selection).flatten()
                scores.append(text_scores)

            scores_df = pd.DataFrame(scores, columns=selection)
            scores_df['Timestamp'] = timestamps

            plt.figure(figsize=(10, 6))
            for emotion in selection:
                plt.plot(scores_df['Timestamp'], scores_df[emotion], label=emotion)

            plt.xlabel('Timestamp')
            plt.ylabel('Emotion Score')
            plt.title(f'Emotion Changes Over Time for {st.session_state.selected_role} in Relationship ID: {st.session_state.selected_relationship_id}')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)

            st.write("**Detected Emotions Over Time:**")
            st.write(scores_df)
        else:
            st.write("No data available for the selected Relationship ID.")

# if st.button("Perform Emotion Detection"):
    df['Response Datetime'] = pd.to_datetime(df['Response Datetime'])
    df['Year'] = df['Response Datetime'].dt.year

if view_format == "View combined/average moods of all users together":
    yearly_texts = aggregate_text_by_year(df, text_column, 'Response Datetime')

    yearly_emotion_scores = {year: classify_text([text], selection).flatten() for year, text in zip(yearly_texts['Year'], yearly_texts[text_column])}

    emotion_df = pd.DataFrame.from_dict(yearly_emotion_scores, orient='index', columns=selection)

    st.write("**Detected Emotions of all users by year:**")
    st.write(emotion_df)

    plt.figure(figsize=(10, 6))
    for emotion in selection:
        plt.plot(emotion_df.index, emotion_df[emotion], label=emotion)

    plt.xlabel('Year')
    plt.ylabel('Score')
    plt.title('Emotion Detection Over the Years')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

elif view_format == "View each mood individually (get the moods of all users in a table view)":

    mentee_sample_data = df[df['Mentor'].isin(['Mentee', 'Mentor'])]
    mentee_sample_data_list = mentee_sample_data[text_column].astype(str).tolist()
    mentee_timestamps_list = mentee_sample_data['Response Datetime'].tolist()

    mentee_filtered_data = [
        (text, timestamp) for text, timestamp in zip(mentee_sample_data_list, mentee_timestamps_list)
        if len(text.split()) > 5
    ]

    if mentee_filtered_data:
        mentee_filtered_texts_list, mentee_filtered_timestamps_list = zip(*mentee_filtered_data)
    else:
        mentee_filtered_texts_list, mentee_filtered_timestamps_list = [], []

    predicted_labels = []
    predicted_scores = []

    for text in mentee_filtered_texts_list:
        scores = classify_text([text], selection)
        scores = scores.flatten()
        top_scores = sorted(zip(selection, scores), key=lambda x: x[1], reverse=True)[:5]
        predicted_labels.append([label for label, score in top_scores])
        predicted_scores.append([score for label, score in top_scores])

    data = {
        'Timestamp': mentee_filtered_timestamps_list,
        'Response': mentee_filtered_texts_list,
        'Predicted Labels': predicted_labels,
        'Scores': predicted_scores
    }

    predicted_df = pd.DataFrame(data)
    
    final_df = pd.merge(mentee_sample_data, predicted_df, left_on=[text_column, 'Response Datetime'], right_on=['Response', 'Timestamp'], how='left')

    pd.set_option('display.max_colwidth', None)

    mentees_emotions_df = final_df[["Mentee ID", "Relationship ID", "Response Datetime", "Predicted Labels", "Scores"]]

    mentees_emotions_df["Response Datetime"] = pd.to_datetime(mentees_emotions_df["Response Datetime"])
    mentees_emotions_df["Month"] = mentees_emotions_df["Response Datetime"].dt.month
    mentees_emotions_df["Year"] = mentees_emotions_df["Response Datetime"].dt.year

    def mean_of_lists(series):
        return list(pd.DataFrame(series.tolist()).mean())

    grouped_df = mentees_emotions_df.groupby(["Mentee ID", "Relationship ID", "Month", "Year"]).agg({
        "Scores": mean_of_lists
    }).reset_index()

    grouped_df["Predicted Labels"] = mentees_emotions_df.groupby(["Mentee ID", "Relationship ID", "Month", "Year"])["Predicted Labels"].apply(lambda x: list(pd.Series([label for sublist in x if isinstance(sublist, list) for label in sublist]).unique())).reset_index(drop=True)

    grouped_df = grouped_df.sort_values(by=['Year', 'Month', 'Mentee ID', 'Relationship ID'])

    st.write("**Detected Emotions by Mentee and Month:**")
    st.write(grouped_df)

# elif view_format == "Measure how the mood changes over time for a single relationship":
#     relationship_ids = df['Relationship ID'].unique().tolist()
#     selected_relationship_id = st.selectbox("Select or enter a Relationship ID", relationship_ids)

#     view_categories1 = ["Mentee", "Mentor"]
#     selected_role = st.radio('Select viewing format', options=view_categories1)

#     if st.button("Measure Mood Changes Over Time", key='2'):
#         relationship_df = df[df['Relationship ID'] == selected_relationship_id]

#         if selected_role == "Mentee":
#             relationship_df = relationship_df[relationship_df['Mentor'] == 'Mentee']
#         elif selected_role == "Mentor":
#             relationship_df = relationship_df[relationship_df['Mentor'] == 'Mentor']

#         if not relationship_df.empty:
#             relationship_df['Response Datetime'] = pd.to_datetime(relationship_df['Response Datetime'])
#             relationship_df = relationship_df.sort_values(by='Response Datetime')

#             texts = relationship_df[text_column].astype(str).tolist()
#             timestamps = relationship_df['Response Datetime'].tolist()

#             scores = []
#             for text in texts:
#                 text_scores = classify_text([text], selection).flatten()
#                 scores.append(text_scores)

#             scores_df = pd.DataFrame(scores, columns=selection)
#             scores_df['Timestamp'] = timestamps

#             plt.figure(figsize=(10, 6))
#             for emotion in selection:
#                 plt.plot(scores_df['Timestamp'], scores_df[emotion], label=emotion)

#             plt.xlabel('Timestamp')
#             plt.ylabel('Emotion Score')
#             plt.title(f'Emotion Changes Over Time for Relationship ID: {selected_relationship_id}')
#             plt.legend()
#             plt.grid(True)
#             st.pyplot(plt)

#             st.write("**Detected Emotions Over Time:**")
#             st.write(scores_df)
#         else:
#             st.write("No data available for the selected Relationship ID.")
# else:
#     relationship_ids = df['Relationship ID'].unique().tolist()
#     selected_relationship_id = st.selectbox("Select or enter a Relationship ID", relationship_ids)

#     view_categories1 = ["Mentee", "Mentor"]
#     selected_role = st.radio('Select viewing format', options=view_categories1)

#     if st.button("Measure Mood Changes Over Time") or 'mood_analysis' in st.session_state:
#         st.session_state['mood_analysis'] = True
#         relationship_df = df[df['Relationship ID'] == selected_relationship_id]

#         if selected_role == "Mentee":
#             relationship_df = relationship_df[relationship_df['Mentor'] == 'Mentee']
#         elif selected_role == "Mentor":
#             relationship_df = relationship_df[relationship_df['Mentor'] == 'Mentor']

#         if not relationship_df.empty:
#             relationship_df['Response Datetime'] = pd.to_datetime(relationship_df['Response Datetime'])
#             relationship_df = relationship_df.sort_values(by='Response Datetime')

#             texts = relationship_df[text_column].astype(str).tolist()
#             timestamps = relationship_df['Response Datetime'].tolist()

#             scores = []
#             for text in texts:
#                 text_scores = classify_text([text], selection).flatten()
#                 scores.append(text_scores)

#             scores_df = pd.DataFrame(scores, columns=selection)
#             scores_df['Timestamp'] = timestamps

#             plt.figure(figsize=(10, 6))
#             for emotion in selection:
#                 plt.plot(scores_df['Timestamp'], scores_df[emotion], label=emotion)

#             plt.xlabel('Timestamp')
#             plt.ylabel('Emotion Score')
#             plt.title(f'Emotion Changes Over Time for Relationship ID: {selected_relationship_id}')
#             plt.legend()
#             plt.grid(True)
#             st.pyplot(plt)

#             st.write("**Detected Emotions Over Time:**")
#             st.write(scores_df)
#         else:
#             st.write("No data available for the selected Relationship ID.")