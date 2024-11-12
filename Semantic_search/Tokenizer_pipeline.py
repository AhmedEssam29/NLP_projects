import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))


# Load preprocessed data
# Replace these with your actual DataFrames
train_text = pd.read_csv("D:/projects/Semantic_search/splitted_data/train_data.csv")  # Ensure this contains the cleaned_text column
test_text = pd.read_csv("D:/projects/Semantic_search/splitted_data/test_data.csv")
validate_text = pd.read_csv("D:/projects/Semantic_search/splitted_data/valid_data.csv")

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_train = vectorizer.fit_transform(train_text['text'])

# Initialize KNN model
k = 5  # Number of neighbors
knn = NearestNeighbors(n_neighbors=k, metric='cosine')
knn.fit(tfidf_train)

# Streamlit App Interface
st.title("Semantic Search in Articles")
st.write("Enter a piece of text to find similar articles:")
lemmatizer = WordNetLemmatizer()
# User input
user_input = st.text_area("Input Text", "Type something here...")

if st.button("Search"):
    if user_input.strip():
        # Preprocess user input similarly to training data
        def preprocess_input(text):
            # Lowercase
            text = text.lower()
            # Remove URLs
            text = re.sub(r'http\S+|www.\S+', '', text)
            # Remove punctuation and numbers
            text = re.sub(r'[^a-z\s]', '', text)
            # Tokenize, remove stop words, and lemmatize
            words = [lemmatizer.lemmatize(word) for word in word_tokenize(text) if word not in stop_words]
            return ' '.join(words)

        # Preprocess and vectorize the input
        user_input_processed = preprocess_input(user_input)
        user_input_vector = vectorizer.transform([user_input_processed])

        # Find similar articles
        distances, indices = knn.kneighbors(user_input_vector)

        # Display results
        st.write("Top similar articles:")
        for i, (index, dist) in enumerate(zip(indices[0], distances[0])):
            similarity = 1 - dist
            st.write(f"**Result {i + 1} (Similarity: {similarity:.4f})**")
            st.write(train_text['text'].iloc[index][:500] + "...")
            st.write("---")
    else:
        st.write("Please enter some text to search.")
