
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
import chardet
import io

nltk.download('stopwords')

warnings.filterwarnings("ignore", message="A parameter name that contains `beta` will be renamed internally to `bias`")
warnings.filterwarnings("ignore", message="A parameter name that contains `gamma` will be renamed internally to `weight`")

st.set_page_config(
    page_title="User Matching Recommender System",
    page_icon="😊",
    initial_sidebar_state='auto',
)

#  full width dataframe
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
st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

st.title("User Matching Recommender System")
st.write("Upload a CSV file with a user profile in each row.")

if 'df' not in st.session_state:
    st.session_state['df'] = None

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        raw_data = uploaded_file.read()
        detected_encoding = chardet.detect(raw_data)['encoding']
        uploaded_file.seek(0)  # Reset file pointer
        
        valid_lines = []
        with io.StringIO(raw_data.decode(detected_encoding, errors='replace')) as f:
            for line in f:
                try:
                    line.encode('utf-8')
                    valid_lines.append(line) 
                except UnicodeDecodeError:
                    st.warning("Skipped a problematic row due to decoding issues.")

        cleaned_file = io.StringIO("".join(valid_lines))
        df = pd.read_csv(cleaned_file)
        st.session_state['df'] = df

        st.success("File successfully loaded after skipping problematic rows.")
        st.write("Preview of the cleaned data:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error processing the file: {e}")
        st.stop()
else:
    df = st.session_state['df']


# st.title("User Matching Recommender System")
# st.write("""Upload a CSV file with a user profile in each row""")

# if 'uploaded_file' not in st.session_state:
#     st.session_state['uploaded_file'] = None

# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     st.session_state['uploaded_file'] = df
# else:
#     df = st.session_state['uploaded_file']

# if df is not None:
#     st.write("Uploaded CSV file:")
#     st.write(df.head())

#     raw_data = uploaded_file.read()
#     result = chardet.detect(raw_data)
#     detected_encoding = result['encoding']
#     uploaded_file.seek(0)
#     df = pd.read_csv(uploaded_file, encoding=detected_encoding)
#     df = df.astype(str)

#     valid_lines = []
#     with io.StringIO(raw_data.decode(detected_encoding, errors='replace')) as f:
#         for line in f:
#             try:
#                 line.encode('utf-8')
#                 valid_lines.append(line) 
#             except UnicodeDecodeError:
#                 st.warning("Skipped a problematic row due to decoding issues.")

#     cleaned_file = io.StringIO("".join(valid_lines))
#     try:
#         df = pd.read_csv(cleaned_file)
#         st.success("File successfully loaded after skipping problematic rows.")
#     except Exception as e:
#         st.error(f"Failed to load data: {e}")
#         st.stop()

  #  df = df.sample(frac=0.1, random_state=42)

required_columns = ['Id', 'Relationship Role']
if not all(col in df.columns for col in required_columns):
    st.error("The uploaded file must include 'Id' and 'Relationship Role' columns.")


model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


st.write("Processing Data...")
    
    # df['Profile'] = (df['[mentor and mentee] Career Interests'] + ' ' + df['[mentor and mentee] Career Interests'] + ' ' + df['[mentor and mentee] Career Interests'] + ' ' +
    #                  df['[mentor and mentee] Your hobbies'] + ' ' + df['[mentor and mentee] Your hobbies'] + ' ' + df['[mentor and mentee] Your hobbies'] + ' ' +
    #                  df.drop(columns=['Id', 'Created at', 'Relationship Role', 'Total Mentees', 'Number of Messages Sent', 'Resource Clicks', 'Courses Clicks']).agg(' '.join, axis=1))

#  if the DataFrame has sufficient columns
if len(df.columns) < 3:
    st.error("Not enough columns . The file must have at least 3 columns.")
    st.stop()

try:
    df['Profile'] = df.iloc[:, 2:].astype(str).agg(' '.join, axis=1)
    st.success("'Profile' column created successfully.")
except Exception as e:
    st.error(f"An error occurred while creating 'Profile': {e}")
    st.stop()


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
    for j in range(k):  # Changed to include all neighbors
        neighbor_role = df.iloc[indices[i][j]]['Relationship Role']
        current_role = df.iloc[i]['Relationship Role']
        
        #st.write(f"User {df.iloc[i]['Id']} ({current_role}) - Potential neighbor {df.iloc[indices[i][j]]['Id']} ({neighbor_role})")
        
        # Skip self-matches (when j == 0)
        if j == 0:
            continue
            
        if current_role == 'mentor' and neighbor_role == 'mentee':
            nearest_neighbors.append((df.iloc[indices[i][j]]['Id'], round(1 - distances[i][j], 2), neighbor_role))
        elif current_role == 'mentee' and neighbor_role == 'mentor':
            nearest_neighbors.append((df.iloc[indices[i][j]]['Id'], round(1 - distances[i][j], 2), neighbor_role))

        #st.write(f"Found {len(nearest_neighbors)} matches for user {df.iloc[i]['Id']}")
        
        result = {
            'Id': df.iloc[i]['Id'],
            'Relationship Role': current_role,
            'Nearest Neighbors': nearest_neighbors
        }
        results.append(result)

grouped_neighbors = {}

for i, profile in enumerate(df['Profile']):
    current_user_id = df.iloc[i]['Id']
    current_role = df.iloc[i]['Relationship Role']

    #collect neighbors
    neighbors = []
    for j in range(k):
        neighbor_id = df.iloc[indices[i][j]]['Id']
        neighbor_role = df.iloc[indices[i][j]]['Relationship Role']
        similarity = round(1 - distances[i][j], 2)

        # Skip self-matches (when j == 0)
        if j == 0:
            continue

        neighbors.append(f"({neighbor_id}, {neighbor_role}, {similarity})")

    grouped_neighbors[current_user_id] = {
        "ID": current_user_id,
        "Relationship Role": current_role,
        "Neighbors": ", ".join(neighbors) 
    }

neighbor_summary_df = pd.DataFrame(grouped_neighbors.values())

st.write("Neighbor Matching Results (all matches):")
st.dataframe(neighbor_summary_df)

summary_csv = neighbor_summary_df.to_csv(index=False)
st.download_button(
    label="Download Grouped Neighbor Matching (all matches) Results as CSV",
    data=summary_csv,
    file_name='grouped_neighbor_matching_results.csv',
    mime='text/csv',
)

#----------------------------------------

grouped_neighbors = {}

for i, profile in enumerate(df['Profile']):
    current_user_id = df.iloc[i]['Id']
    current_role = df.iloc[i]['Relationship Role']
    #current_role = current_role.replace('mentor', 'Mentor').replace('mentee', 'Mentee')

    neighbors = []
    for j in range(k):
        neighbor_id = df.iloc[indices[i][j]]['Id']
        neighbor_role = df.iloc[indices[i][j]]['Relationship Role']
        similarity = round(1 - distances[i][j], 2)

        if j == 0:
            continue

        if (current_role.lower() == 'mentor' and neighbor_role.lower() == 'mentee') or \
           (current_role.lower() == 'mentee' and neighbor_role.lower() == 'mentor'):
            neighbors.append(f"({neighbor_id}, {neighbor_role}, {similarity})")

    grouped_neighbors[current_user_id] = {
        "ID": current_user_id,
        "Relationship Role": current_role,
        "Neighbors": ", ".join(neighbors) 
    }

neighbor_summary_df2 = pd.DataFrame(grouped_neighbors.values())

st.write("Neighbor Matching Results (mentees to mentors only):")
st.dataframe(neighbor_summary_df2)

summary_csv = neighbor_summary_df2.to_csv(index=False)
st.download_button(
    label="Download Grouped Neighbor Matching (mentees to mentors only) Results as CSV",
    data=summary_csv,
    file_name='neighbor_matching_results.csv',
    mime='text/csv',
)

#table of role breakdown
df['Relationship Role'] = df['Relationship Role'].replace({'mentor': 'Mentor', 'mentee': 'Mentee'})

role_counts = df['Relationship Role'].value_counts().reset_index()
role_counts.columns = ['Relationship Role', 'Count']

st.write("Breakdown of Relationship Roles:")
st.dataframe(role_counts)

