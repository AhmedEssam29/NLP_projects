import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import torch

# Download NLTK data (if not already downloaded)
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

# Preprocess function
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Load Data and Preprocess
train_df = pd.read_csv("D:/projects/Semantic_search/splitted_data/train_data.csv")  # Adjust path
train_df['text'] = train_df['text'].apply(preprocess_text)

# Load Sentence Transformer Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Generate embeddings for each text entry, and convert them to NumPy arrays on CPU
embeddings = model.encode(train_df['text'].tolist(), convert_to_tensor=True).cpu().numpy()

# Initialize FAISS Index
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# Streamlit Interface
st.title("Semantic Search in Articles with Transformer Embeddings and FAISS")
st.write("Enter a piece of text to find similar articles:")

# User input
user_input = st.text_area("Input Text", "Type something here...")

if st.button("Search"):
    if user_input.strip():
        # Preprocess and Embed User Input
        user_input_processed = preprocess_text(user_input)
        query_embedding = model.encode([user_input_processed], convert_to_tensor=True).cpu().numpy()

        # Perform Search
        k = 5  # Number of results
        distances, indices = faiss_index.search(query_embedding, k)

        # Display results
        st.write("Top similar articles:")
        for i, (index, dist) in enumerate(zip(indices[0], distances[0])):
            similarity = 1 - dist  # Convert distance to similarity
            st.write(f"**Result {i + 1} (Similarity: {similarity:.4f})**")
            st.write(train_df['text'].iloc[index][:500] + "...")
            st.write("---")
    else:
        st.write("Please enter some text to search.")
