import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sys 
from sklearn.manifold import TSNE
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    if isinstance(text, str):
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
        return tokens
    else:
        return []

# Function to create word cloud
def create_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)

# Function to visualize topics using Plotly
def visualize_topics_plotly(lda_model):
    topics = lda_model.show_topics(formatted=False, num_words=10)
    topic_words = []
    for i, topic in enumerate(topics):
        topic_words_list = [word for word, _ in topic[1]]
        topic_words.append(", ".join(topic_words_list))
    
    fig = px.bar(x=list(range(1, len(topic_words)+1)), y=[1]*len(topic_words), 
                 hover_data={'x': list(range(1, len(topic_words)+1)), 'y': topic_words},
                 labels={'x': 'Topic', 'y': 'Top Words'}, 
                 title='Topic Distribution')
    
    return fig


def get_document_topics_matrix(lda_model, corpus, num_topics):
    """Convert document-topic distributions to a matrix format."""
    doc_topics = []
    for doc in corpus:
        topic_probs = [0] * num_topics
        for topic, prob in lda_model.get_document_topics(doc):
            topic_probs[topic] = prob
        doc_topics.append(topic_probs)
    return np.array(doc_topics)

def visualize_topic_clusters(lda_model, corpus, num_topics, df, text_column):
    """Create an interactive scatter plot of document clusters."""
    try:
        # Get document-topic distribution matrix
        doc_topics_matrix = get_document_topics_matrix(lda_model, corpus, num_topics)
        
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        doc_topics_2d = tsne.fit_transform(doc_topics_matrix)
        
        # Get dominant topics
        dominant_topics = []
        for doc in corpus:
            topic_probs = lda_model.get_document_topics(doc)
            dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
            dominant_topics.append(f"Topic {dominant_topic + 1}")
        
        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'x': doc_topics_2d[:, 0],
            'y': doc_topics_2d[:, 1],
            'Dominant_Topic': dominant_topics
            #'Original_Text': df[text_column].values
        })
        
        # Create the scatter plot
        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            color='Dominant_Topic',
            hover_data=['Original_Text'],
            title='Topic Clusters Visualization (t-SNE)',
            labels={'Dominant_Topic': 'Main Topic'},
        )
        
        # Update the layout
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(
            plot_bgcolor='white',
            width=800,
            height=600,
            xaxis_title="t-SNE dimension 1",
            yaxis_title="t-SNE dimension 2"
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error in visualization: {str(e)}")
        return None
    
# Title and description
st.title("NLP Preprocessing and Topic Modeling App")
st.write("""
Upload a CSV file, select a text column for NLP preprocessing,
and then wait for the model to perform topic modeling to visualize the topics using LDA.
""")

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

    # Select text column
    text_column = st.selectbox("Select the text column for preprocessing", df.columns)

    # Preprocess text
    df['processed_text'] = df[text_column].apply(preprocess_text)

    # Show preprocessed text
    st.write("Preprocessed Text:")
    st.write(df[['processed_text']].head())

    # Filter out rows with empty or NaN processed_text
    df = df.dropna(subset=['processed_text'])
    df = df[df['processed_text'].apply(len) > 0]

    # Create dictionary and corpus for LDA
    dictionary = corpora.Dictionary(df['processed_text'])
    corpus = [dictionary.doc2bow(text) for text in df['processed_text']]

    # LDA model
    num_topics = st.slider("Select number of topics", 2, 50, 5)
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)


    # Topic clustering visualization
    st.write("### Topic Clusters Visualization")
    st.write("This visualization shows how documents cluster together based on their topic distributions. " 
                "Documents with similar topic words appear closer together.")
    
    clustering_fig = visualize_topic_clusters(lda_model, corpus, num_topics, df, text_column)
    if clustering_fig:
        st.plotly_chart(clustering_fig, use_container_width=True)

    # Show topics
    st.write("LDA Topics:")
    topics = lda_model.print_topics(num_words=10)
    for topic in topics:
        st.write(topic)

    # Visualize topics with Plotly
    st.write("Topic Visualization:")
    fig = visualize_topics_plotly(lda_model)
    st.plotly_chart(fig, use_container_width=True)


    # Search for topics
    search_word = st.text_input("Enter a word to find related topics:")
    if search_word:
        search_topics = lda_model.get_term_topics(search_word)
        search_results = pd.DataFrame(search_topics, columns=['Topic', 'Probability']).sort_values(by='Probability', ascending=False)
        st.write(f"Topics related to '{search_word}':")
        st.write(search_results)

    # Visualize topics with word clouds
    for i, topic in enumerate(lda_model.show_topics(formatted=False, num_words=10)):
        st.write(f"Topic {i+1}")
        words = [word for word, _ in topic[1]]
        create_wordcloud(words, f"Topic {i+1}")

for i in range(4000):
    try:
        print(i, flush=True)
    except BrokenPipeError:
        sys.stdout = None