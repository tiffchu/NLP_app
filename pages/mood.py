# import pandas as pd
# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# import numpy as np

# st.set_page_config(
#     page_title="Emotion Detection App",
#     page_icon="😊",
#     initial_sidebar_state='auto',
# )

# st.title("Emotion Detection with BART")
# st.write("""
# Upload a CSV file, select a text column for NLP preprocessing,
# and perform emotion detection.
# """)

# if 'uploaded_file' not in st.session_state:
#     st.session_state['uploaded_file'] = None

# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.session_state['uploaded_file'] = df  # Save the DataFrame to session state
# else:
#     df = st.session_state['uploaded_file']

# if df is not None:
#     st.write("Uploaded CSV file:")
#     st.write(df.head())

#     text_column = st.selectbox("Select the text column for emotion detection", df.columns)

#     model_name = "facebook/bart-large-mnli"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)

#     mood_categories = [
#         "happiness", "excitement", "gratitude", "contempt", "sadness", "nervousness",
#         "stress", "boredom", "anger", "fear", "surprise", "disgust", "calmness",
#         "curiosity", "envy", "guilt", "shame", "pride", "love", "loneliness",
#         "hope", "despair", "relief", "anticipation", "trust", "confusion",
#         "frustration", "embarrassment", "amusement", "jealousy"
#     ]

#     selection = st.multiselect('Select a range of emotions to predict (3 to 9)', options=mood_categories)

#     if "Add emotion" in selection:
#         custom_emotion = st.text_input("Enter your custom emotion")
#         if custom_emotion:
#             selection.append(custom_emotion)

#     def classify_text(texts, labels):
#         inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
#         outputs = model(**inputs)
#         logits = outputs.logits.detach().cpu().numpy()
#         scores = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
#         return scores

#     if st.button("Perform Emotion Detection"):
#         mentee_sample_data = df[df['Mentor'] == 'Mentee']
#         mentee_sample_data_list = mentee_sample_data[text_column].tolist()
#         mentee_timestamps_list = mentee_sample_data['Response Datetime'].tolist()

#         mentee_filtered_data = [
#             (str(text), timestamp) for text, timestamp in zip(mentee_sample_data_list, mentee_timestamps_list)
#             if len(str(text).split()) > 5
#         ]

#         if mentee_filtered_data:
#             mentee_filtered_texts_list, mentee_filtered_timestamps_list = zip(*mentee_filtered_data)
#         else:
#             mentee_filtered_texts_list, mentee_filtered_timestamps_list = [], []

#         predicted_labels = []
#         predicted_scores = []

#         for text, timestamp in zip(mentee_filtered_texts_list, mentee_filtered_timestamps_list):
#             scores = classify_text([text], selection)
#             scores = scores.flatten()
#             top_scores = sorted(zip(selection, scores), key=lambda x: x[1], reverse=True)[:5]
#             predicted_labels.append([label for label, score in top_scores])
#             predicted_scores.append([score for label, score in top_scores])

#         data = {
#             'Timestamp': mentee_filtered_timestamps_list,
#             'Response': mentee_filtered_texts_list,
#             'Predicted Labels': predicted_labels,
#             'Scores': predicted_scores
#         }

#         predicted_df = pd.DataFrame(data)
#         final_df = pd.merge(df, predicted_df, on=text_column, how='inner')

#         pd.set_option('display.max_colwidth', None)

#         mentees_emotions_df = final_df[["Mentee ID", "Relationship ID", "Response Datetime", "Predicted Labels", "Scores"]]
#         mentees_emotions_df = pd.DataFrame(mentees_emotions_df)

#         mentees_emotions_df["Response Datetime"] = pd.to_datetime(mentees_emotions_df["Response Datetime"])

#         mentees_emotions_df["Month"] = mentees_emotions_df["Response Datetime"].dt.month
#         mentees_emotions_df["Year"] = mentees_emotions_df["Response Datetime"].dt.year

#         def mean_of_lists(series):
#             return list(pd.DataFrame(series.tolist()).mean())

#         grouped_df = mentees_emotions_df.groupby(["Mentee ID", "Relationship ID", "Month", "Year"]).agg({
#             "Scores": mean_of_lists
#         }).reset_index()

#         grouped_df["Predicted Labels"] = mentees_emotions_df.groupby(["Mentee ID", "Relationship ID", "Month", "Year"])["Predicted Labels"].apply(lambda x: list(pd.Series([label for sublist in x for label in sublist]).unique())).reset_index(drop=True)

#         grouped_df = grouped_df.sort_values(by=['Year', 'Month', 'Mentee ID', 'Relationship ID'])

#         st.write("**Detected Emotions by Mentee and Month:**")
#         st.write(grouped_df)


#         ###
#         view_categories = ["view each mentee's mood individually", "view all mentees' moods collectively"]

#         selection2 = st.multiselect('Select viewing format', options=view_categories)
#         if selection2 == "view all mentees' moods collectively":

#             st.write("Mood Over Time")
#             month_column = st.selectbox("Select month column for time series analysis", grouped_df.columns)

            
#             #grouped_df2 = grouped_df.groupby('')[''].apply(' '.join).reset_index()

#             if month_column and month_column in grouped_df.columns:
#                         st.write("Calculating word count over time...")
#                         df[month_column] = pd.to_datetime(df[month_column])
#                         moodgraph = st.selectbox("Select mood for time series line graph", mood_categories)

#                         if moodgraph and moodgraph in df.columns:
                             

#                         ##
#                         # df['Word Count'] = df[moodgraph].apply(count_words)
#                         # df['DateGroup'] = df[month_column].dt.to_period('Q')
#                         # word_count_over_time = df.groupby('DateGroup')['Word Count'].sum()
#                         # fig, ax = plt.subplots()
#                         # ax.plot(word_count_over_time.index.to_timestamp(), word_count_over_time.values, marker='o', linestyle='-')
#                         # ax.set_xlabel("Date")
#                         # ax.set_ylabel('Total Word Count')
#                         # ax.set_title('Total Word Count Over Time (3-month periods)')
#                         # plt.xticks(rotation=70)
#                         # st.pyplot(fig)
#                         # 





