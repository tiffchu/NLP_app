import pandas as pd
import streamlit as st 
from transformers import pipeline


st.title("Emotion Detection")
st.write("""
Upload a CSV file, select a text column for NLP preprocessing,
and perform mood detection.
""")

if 'uploaded_file' in st.session_state:
    uploaded_file = st.session_state['uploaded_file']
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

    columns = df.columns.tolist()

    text_column = st.selectbox("Select the text column for mood detection", df.columns)

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


    mood_categories = ["happiness", "excitement", "gratitude", "contempt", "sadness", "nervousness", "stress", "boredom", "anger", 
                       "fear", "surprise", "disgust", "calmness", "curiosity", "envy", "guilt", "shame", "pride", "love", "loneliness", 
                       "hope", "despair", "relief", "anticipation", "trust", "confusion", "frustration", "embarrassment", "amusement",
                        "jealousy"]  + ["Add emotion"]



    #st.write('Select a range of emotions that youd like to predict (3 to 9)') #, mood_categories)

    selection = st.multiselect('Select a range of emotions that youd like to predict (3 to 9)', options=mood_categories)
    
    # if selection == "Add Emotion":
    #     st.text_input("Enter your other option")

    if "Add emotion" in selection:
        custom_emotion = st.text_input("Enter your custom emotion")
        if custom_emotion is not None:
            selection.append(custom_emotion)

data1 = df[df['Mentor'] == 'Mentee'].iloc[1000:1300]

data_list = data1['Response'].tolist()
mentee_timestamps_list = data1['Response Datetime'].tolist()

# Convert all entries to strings and filter out text segments with fewer than 5 words
mentee_filtered_data = [
    (str(text), timestamp) for text, timestamp in zip(data_list, mentee_timestamps_list)
    if len(str(text).split()) > 5
]

if mentee_filtered_data: 
    mentee_filtered_texts_list, mentee_filtered_timestamps_list = zip(*mentee_filtered_data)
else:
    mentee_filtered_texts_list, mentee_filtered_timestamps_list = [], []

threshold = 0.0

predicted_labels = []
predicted_scores = []
results = []

for text, timestamp in zip(mentee_filtered_texts_list, mentee_filtered_timestamps_list):
    result = classifier(text, candidate_labels=mood_categories)
    results.append((result, timestamp))
    
    labels_above_threshold = [(label, round(score, 6)) for label, score in zip(result['labels'], result['scores']) if score >= threshold]
    predicted_labels.append([label for label, score in labels_above_threshold])
    predicted_scores.append([score for label, score in labels_above_threshold])


data = {
    'Timestamp': mentee_filtered_timestamps_list,
    'Response': mentee_filtered_texts_list,
    'Predicted Labels': predicted_labels,
    'Scores': predicted_scores
}

predicted_df = pd.DataFrame(data)

final_df = pd.merge(df, predicted_df, on='Response', how='inner')

pd.set_option('display.max_colwidth', None)
