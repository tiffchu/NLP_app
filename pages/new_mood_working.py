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

    selection = st.multiselect('Select emotions to predict', options=mood_categories)

    if "Add emotion" in selection:
        selection.remove("Add emotion")
        custom_emotion = st.text_input("Enter your custom emotion")
        if custom_emotion:
            selection.append(custom_emotion)

    @st.cache_data
    def classify_text(texts, labels, batch_size=16):
        all_scores = []
        
        for i in range(0, len(texts) if isinstance(texts, list) else 1, batch_size):
            batch_texts = texts[i:i + batch_size] if isinstance(texts, list) else [texts]
            
            hypotheses = []
            for text in batch_texts:
                for label in labels:
                    hypotheses.append(f"This text expresses {label}.")
            
            premise_texts = [text for text in batch_texts for _ in labels]
            inputs = tokenizer(premise_texts, hypotheses, padding=True, truncation=True, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                probs = torch.softmax(logits, dim=1)[:, 2]
                
                batch_scores = probs.reshape(-1, len(labels))
                all_scores.extend(batch_scores.tolist())
        
        return np.array(all_scores)

    def aggregate_text_by_year(dataframe, text_col, date_col):
        dataframe[date_col] = pd.to_datetime(dataframe[date_col])
        dataframe['Year'] = dataframe[date_col].dt.year
        dataframe[text_col] = dataframe[text_col].astype(str).fillna('')
        return dataframe.groupby('Year')[text_col].apply(' '.join).reset_index()

    view_categories = [
        "View each mood individually (get the moods of all users in a table view)",
        "View combined/average moods of all users together",
        "Measure how the mood changes over time for a single relationship"
    ]
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

        st.session_state.selected_role = st.radio('Select role', options=["Mentee", "Mentor"])

        if st.button("Measure Mood Changes Over Time", key='1'):
            relationship_df = df[df['Relationship ID'] == st.session_state.selected_relationship_id]
                        
            role_filter = 'Mentee' if st.session_state.selected_role == "Mentee" else 'Mentor'
            relationship_df = relationship_df[relationship_df['Mentor'] == role_filter]

            if not relationship_df.empty and selection:
                # Sort by Response Datetime and create response numbers
                relationship_df = relationship_df.sort_values(by='Response Datetime')
                relationship_df['Response_Number'] = range(1, len(relationship_df) + 1)

                texts = relationship_df[text_column].astype(str).tolist()
                response_numbers = relationship_df['Response_Number'].tolist()

                with st.spinner('Analyzing emotions...'):
                    scores = classify_text(texts, selection)
                    scores_df = pd.DataFrame(scores, columns=selection)
                    scores_df['Response_Number'] = response_numbers

                    n_emotions = len(selection)
                    n_cols = min(3, n_emotions)
                    n_rows = (n_emotions + n_cols - 1) // n_cols
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                    axes = np.array(axes).reshape(-1) if n_emotions > 1 else [axes]

                    for idx, emotion in enumerate(selection):
                        ax = axes[idx]
                        ax.plot(scores_df['Response_Number'], scores_df[emotion], 
                            label=emotion, marker='o')
                        ax.set_title(f'{emotion} Over Responses')
                        ax.set_xlabel('Response Number')
                        ax.set_ylabel('Score')
                        ax.grid(True)

                    plt.tight_layout()
                    st.pyplot(fig)

                    st.write("**Detailed Emotion Scores:**")
                    st.dataframe(scores_df)

    elif view_format == "View combined/average moods of all users together":
        if selection:
            yearly_texts = aggregate_text_by_year(df, text_column, 'Response Datetime')
            
            with st.spinner('Analyzing emotions...'):
                yearly_emotion_scores = {
                    year: classify_text(text, selection).flatten() 
                    for year, text in zip(yearly_texts['Year'], yearly_texts[text_column])
                }

                emotion_df = pd.DataFrame.from_dict(yearly_emotion_scores, orient='index', columns=selection)
                
                st.write("**Average Emotions by Year:**")
                st.dataframe(emotion_df)

                n_emotions = len(selection)
                n_cols = min(3, n_emotions)
                n_rows = (n_emotions + n_cols - 1) // n_cols
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                axes = np.array(axes).reshape(-1) if n_emotions > 1 else [axes]

                for idx, emotion in enumerate(selection):
                    ax = axes[idx]
                    ax.plot(emotion_df.index, emotion_df[emotion], marker='o')
                    ax.set_title(f'{emotion} Over Years')
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Score')
                    ax.grid(True)

                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.warning("Please select at least one emotion to analyze.")

    elif view_format == "View each mood individually":
        if selection:
            with st.spinner('Analyzing individual responses...'):
                mentee_sample_data = df[df['Mentor'].isin(['Mentee', 'Mentor'])]
                texts = mentee_sample_data[text_column].astype(str).tolist()
                
                valid_indices = [i for i, text in enumerate(texts) if len(text.split()) > 5]
                filtered_texts = [texts[i] for i in valid_indices]
                
                if filtered_texts:
                    scores = classify_text(filtered_texts, selection)
                    
                    results_df = pd.DataFrame({
                        'Timestamp': mentee_sample_data['Response Datetime'].iloc[valid_indices],
                        'Response': filtered_texts,
                        'Mentor/Mentee': mentee_sample_data['Mentor'].iloc[valid_indices],
                        'Relationship ID': mentee_sample_data['Relationship ID'].iloc[valid_indices]
                    })
                    
                    for i, emotion in enumerate(selection):
                        results_df[f'Score_{emotion}'] = scores[:, i]
                    
                    st.write("**Individual Response Analysis:**")
                    st.dataframe(results_df)
                else:
                    st.warning("No valid responses found after filtering.")
        else:
            st.warning("Please select at least one emotion to analyze.")