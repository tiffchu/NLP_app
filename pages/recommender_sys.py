# import pandas as pd
# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt

# st.set_page_config(
#     page_title="User Matching Recommender System",
#     page_icon="😊",
#     initial_sidebar_state='auto',
# )

# st.title("User Matching Recommender System")
# st.write("""
# Upload a CSV file with a user profile in each row
# """)

# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.session_state['uploaded_file'] = df
# else:
#     df = st.session_state['uploaded_file']

# if df is not None:
#     st.write("Uploaded CSV file:")
#     st.write(df.head())

import streamlit as st
import warnings
from nltk.corpus import stopwords
import nltk
from transformers import BertModel, BertTokenizer
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

nltk.download('stopwords')

warnings.filterwarnings("ignore", message="A parameter name that contains `beta` will be renamed internally to `bias`")
warnings.filterwarnings("ignore", message="A parameter name that contains `gamma` will be renamed internally to `weight`")

st.set_page_config(
    page_title="User Matching Recommender System",
    page_icon="😊",
    initial_sidebar_state='auto',
)
hide_dataframe_row_index = """
            <style>
            .css-1uixr1i.e16nr0p30 {
                width: 100%;
            }
            .css-1dp5vir.e1fqkh3o3 {
                width: 100%;
            }
            </style>
            """
#    CSS with Markdown
st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

st.title("User Matching Recommender System")
st.write("""Upload a CSV file with a user profile in each row""")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='utf-8')
    df = df.astype(str)
    df = df.head(500)
    df = df.sample(frac=0.1, random_state=42)

    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    st.write("Uploaded File:")
    st.dataframe(df.head())

    st.write("Processing Data...")
    
    df['Profile'] = (df['[mentor and mentee] Career Interests'] + ' ' + df['[mentor and mentee] Career Interests'] + ' ' + df['[mentor and mentee] Career Interests'] + ' ' +
                     df['[mentor and mentee] Your hobbies'] + ' ' + df['[mentor and mentee] Your hobbies'] + ' ' + df['[mentor and mentee] Your hobbies'] + ' ' +
                     df.drop(columns=['Id', 'Created at', 'Relationship Role', 'Total Mentees', 'Number of Messages Sent', 'Resource Clicks', 'Courses Clicks']).agg(' '.join, axis=1))
#TODO TODO TODO  change the above so weight of cols can be adjusted^
    stop_words = set(stopwords.words('english'))
    df['Profile'] = df['Profile'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Profile'])

    inputs = df['Profile'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=512))

    inputs = inputs.tolist()
    max_len = max(len(seq) for seq in inputs)
    padded_inputs = [seq + [0] * (max_len - len(seq)) for seq in inputs]

    input_ids = torch.tensor(padded_inputs)

    with torch.no_grad():
        outputs = model(input_ids)
        embeddings = outputs.last_hidden_state

    embeddings = embeddings.mean(dim=1).numpy()

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    combined_features = pd.concat([pd.DataFrame(embeddings_scaled), pd.DataFrame(tfidf_matrix.toarray())], axis=1)

    svd = TruncatedSVD(n_components=100)
    X_reduced = svd.fit_transform(combined_features)

    cos_sim_matrix = cosine_similarity(X_reduced)

    k = st.slider("Select number of nearest neighbors", 1, 20, 5)
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(X_reduced)

    distances, indices = knn.kneighbors(X_reduced)

    results = []
    for i, profile in enumerate(df['Profile']):
        nearest_neighbors = []
        for j in range(1, k):
            neighbor_role = df.iloc[indices[i][j]]['Relationship Role']
            if df.iloc[i]['Relationship Role'] == 'mentor' and neighbor_role == 'mentee':
                nearest_neighbors.append((df.iloc[indices[i][j]]['Id'], round(1 - distances[i][j], 2), neighbor_role))
            elif df.iloc[i]['Relationship Role'] == 'mentee' and neighbor_role == 'mentor':
                nearest_neighbors.append((df.iloc[indices[i][j]]['Id'], round(1 - distances[i][j], 2), neighbor_role))

        result = {
            'Id': df.iloc[i]['Id'],
            'Relationship Role': df.iloc[i]['Relationship Role'],
            'Nearest Neighbors': nearest_neighbors[:k-1]
        }
        results.append(result)

results_df = pd.DataFrame(results)
pd.set_option('display.max_colwidth', None)
#change Neighbors to string for serialization
results_df['Nearest Neighbors'] = results_df['Nearest Neighbors'].apply(lambda x: str(x) if isinstance(x, list) else x)

st.write("Matching Results:")
st.dataframe(results_df)

csv = results_df.to_csv(index=False)
st.download_button(
    label="Download Matching Results as CSV",
    data=csv,
    file_name='mentor_mentee_matching_results.csv',
    mime='text/csv',
)
