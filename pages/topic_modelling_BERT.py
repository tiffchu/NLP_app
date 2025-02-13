import streamlit as st
import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
# from wordcloud import WordCloud
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer

# from pycaret.nlp import create_model, evaluate_model

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

if 'topic_visualization' not in st.session_state:
    st.session_state.topic_visualization = None

if 'data' not in st.session_state:
    st.session_state.data = None
    
if 'text_column' not in st.session_state:
    st.session_state.text_column = None
    
if 'topic_model' not in st.session_state:
    st.session_state.topic_model = None
    
if 'search_word' not in st.session_state:
    st.session_state.search_word = None

def update_dataset(*, data):
    st.session_state.data = data

def select_text_column(*, column):
    st.session_state.text_column = column
    documents = st.session_state.data[st.session_state.text_column].astype(str)

    # Visualize topics

    st.session_state.topic_model = get_BERTopic_model()
    st.session_state.topic_model.fit_transform(documents)
    st.session_state.topic_visualization = st.session_state.topic_model.visualize_topics()
    

def update_search_word(*, word):
    st.session_state.search_word = word

# preprocess text
def preprocess_text(text):
    if isinstance(text, str):  # Check if the input is a string
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
        return tokens
    else:
        return []  # Return an empty list if the input is not a string

@st.cache_data
def get_BERTopic_model():
    # 1Extract embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 2Reduce dimensionality
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=69)

    # 3 Cluster reduced embeddings
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

    # 4 Tokenize topics
    vectorizer_model = CountVectorizer(stop_words="english")

    # 5 Create topic representation
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    #  6  Fine-tune topic representations with  a `bertopic.representation` model
    representation_model = KeyBERTInspired()

    return BERTopic(
        embedding_model=embedding_model,          # Step 1 - Extract embeddings
        umap_model=umap_model,                    # Step 2 - Reduce dimensionality
        hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
        vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
        ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
        representation_model=representation_model, # Step 6 - (Optional) Fine-tune topic represenations
        language="english",
        top_n_words=10,
        n_gram_range = (1,3)
    )

st.title("NLP Preprocessing and Topic Modeling App")
st.write("""
Upload a CSV file, select a text column for NLP preprocessing,
and perform topic modeling to visualize the topics using BERTopic.
""")

raw_data = st.file_uploader("Choose a CSV file", type="csv")
if raw_data:
    data = pd.read_csv(raw_data)
    st.write("Uploaded CSV file:")
    st.write(data)

    text_column = st.selectbox("Select the column for topic modeling", data.columns)
    if text_column:
        documents = data[text_column].astype(str)
        # Check if enough documents
        valid_docs = [doc for doc in documents if len(doc.strip()) > 0]
        
        if len(valid_docs) >= 15:  # BERTopic needs at least 15 documents by default
            topic_model = get_BERTopic_model()
            topics, probs = topic_model.fit_transform(valid_docs)
            
            freq = topic_model.get_topic_info()
            st.write(freq.head(5))
            
            #Visualize 
            topic_viz = topic_model.visualize_topics()
            st.write("Topic Visualization:")
            st.plotly_chart(topic_viz)
        else:
            st.warning("Please provide at least 15 documents with content for topic modeling")

        #Search functionality
        search_word = st.text_input("Enter a word to find related topics:")
        if search_word:
            search_topics, search_probabilities = topic_model.find_topics(search_word)
            doc_info = topic_model.get_document_info(documents)
            topic_words = [doc_info[doc_info['Topic'] == i]['Name'].iloc[0].split('_')[1:] for i in search_topics]
            
            search_results = pd.DataFrame({
                'Topic': search_topics,
                'Topic Words': topic_words,
                'Probability': search_probabilities
            }).sort_values(by='Probability', ascending=False)
            search_results = search_results[search_results['Topic'] >= 0]
            
            st.write(f"Topics related to '{search_word}':")
            st.write(search_results)
