import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
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

# Function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):  # Check if the input is a string
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
        return tokens
    else:
        return []  # Return an empty list if the input is not a string

# Function to create word cloud
def create_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)

def get_BERTopic_model():
    # Step 1 - Extract embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 2 - Reduce dimensionality
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')


    # Step 3 - Cluster reduced embeddings
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

    # Step 4 - Tokenize topics
    vectorizer_model = CountVectorizer(stop_words="english")

    # Step 5 - Create topic representation
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    # Step 6 - (Optional) Fine-tune topic representations with 
    # a `bertopic.representation` model
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

# Title and description
st.title("NLP Preprocessing and Topic Modeling App")
st.write("""
Upload a CSV file, select a text column for NLP preprocessing,
and perform topic modeling to visualize the topics using BERTopic.
""")

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

    # Select text column
    text_column = st.selectbox("Select the column for topic modeling", df.columns)

    # Preprocess text
    # df['processed_text'] = df[text_column].apply(preprocess_text)

    # # Show preprocessed text
    # st.write("Preprocessed Text:")
    # st.write(df[['processed_text']].head())

    # Filter out rows with empty or NaN processed_text
    # df = df.dropna(subset=['processed_text'])

    # Prepare text for BERTopic
    documents = df[text_column].astype(str)

    # Create BERTopic model
    topic_model = get_BERTopic_model()
    topics, probabilities = topic_model.fit_transform(documents)
    # fileName = st.text_input('File Name')
    # st.download_button('Save Topic Model', topic_model, )

    # Show topics
    #st.write("BERTopic Topics:")
    #st.write(topic_model.get_topics())

    # Visualize topics
    st.write("Topic Visualization:")
    st.plotly_chart(topic_model.visualize_topics(), use_container_width=True)

 # Search for topics
    search_word = st.text_input("Enter a word to find related topics:")
    if search_word:
        search_topics, search_probabilities = topic_model.find_topics(search_word)
        print(search_word)
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
        # st.write(doc_info)

    # # Calculate coherence score
    # dictionary = Dictionary(df['processed_text'])
    # corpus = [dictionary.doc2bow(text) for text in df['processed_text']]
    # coherence_model = CoherenceModel(topics=[topic_model.get_topic(i) for i in range(num_topics)], texts=df['processed_text'], dictionary=dictionary, coherence='c_v')
    # coherence_score = coherence_model.get_coherence()
    # st.write(f"Coherence Score: {coherence_score}")

    # Perplexity score calculation using sklearn's CountVectorizer
    vectorizer = CountVectorizer()
    transformed_documents = vectorizer.fit_transform(documents)



    # # Visualize topics with word clouds
    # for topic_id in range(topic_model.get_number_of_topics()):
    #     st.write(f"Topic {topic_id + 1}")
    #     words = topic_model.get_topic(topic_id)[:10]  # Get top 10 words per topic
    #     create_wordcloud([word for word, _ in words], f"Topic {topic_id + 1}")

    # num_topics = len(set(topics))  # Determine the number of unique topics
    # for topic_id in range(num_topics):
    #     st.write(f"Topic {topic_id + 1}")
    #     words = topic_model.get_topic(topic_id)[:10]  # Get top 10 words per topic
    #     create_wordcloud([word for word, _ in words], f"Topic {topic_id + 1}")


    # Calculate coherence score
    # dictionary = Dictionary(df['processed_text'])
    # corpus = [dictionary.doc2bow(text) for text in df['processed_text']]
    # coherence_model = CoherenceModel(topics=topic_model.get_topics(), texts=df['processed_text'], dictionary=dictionary, coherence='c_v')
    # coherence_score = coherence_model.get_coherence()

    # st.write(f"Coherence Score: {coherence_score}")

    # Perplexity score calculation using sklearn's CountVectorizer
    # vectorizer = CountVectorizer()
    # transformed_documents = vectorizer.fit_transform(documents)
    # perplexity_score = topic_model.perplexity(transformed_documents)

    # st.write(f"Perplexity Score: {perplexity_score}")

    # Search for topics
    search_word = st.text_input("Enter a word to find related topics:")
    if search_word:
        search_topics, search_probabilities = topic_model.find_topics(search_word)
        search_results = pd.DataFrame({
            'Topic': search_topics,
            'Probability': search_probabilities
        }).sort_values(by='Probability', ascending=False)
        st.write(f"Topics related to '{search_word}':")
        st.write(search_results)



