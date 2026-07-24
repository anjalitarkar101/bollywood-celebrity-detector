# ==========================================
# predict.py - Bollywood Celebrity Detector
# ==========================================

import os
import pickle
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import FeatureExtractor


# ==========================================
# Load Functions
# ==========================================

@st.cache_resource
def get_extractor():
    """Load the feature extractor model."""
    return FeatureExtractor(model_name="VGG-Face")


@st.cache_data
def load_data():
    """
    Load celebrity embeddings and filenames.

    Returns:
        tuple: (feature_vectors, filenames)
    """
    try:
        with open('models/embeddings.pkl', 'rb') as f:
            feature_vectors = pickle.load(f)
        with open('models/filenames.pkl', 'rb') as f:
            filenames = pickle.load(f)
        return feature_vectors, filenames
    except FileNotFoundError:
        return None, None


# ==========================================
# Recommendation Function
# ==========================================

def find_celebrity_match(feature_vectors, uploaded_feature_vector, filenames):
    """
    Find the best celebrity match using cosine similarity.

    Args:
        feature_vectors: All celebrity feature vectors
        uploaded_feature_vector: Uploaded image feature vector
        filenames: List of celebrity image paths

    Returns:
        dict: {
            'name': Celebrity name,
            'image_path': Path to celebrity image,
            'similarity': Similarity score
        }
    """
    similarities = []

    # Calculate cosine similarity with each celebrity
    for celebrity_feature_vector in feature_vectors:
        sim = cosine_similarity(
            uploaded_feature_vector.reshape(1, -1),
            celebrity_feature_vector.reshape(1, -1)
        )[0][0]
        similarities.append(sim)

    # Get the best match (highest similarity score)
    top_idx = np.argmax(similarities)

    # Extract actor name from file path
    # Example: data/Aamir_Khan/image1.jpg -> Aamir Khan
    file_path = filenames[top_idx]
    actor_name = file_path.split(os.sep)[1]  # Get folder name
    actor_name = actor_name.replace('_', ' ')  # Replace underscores with spaces

    return {
        'name': actor_name,
        'image_path': file_path,
        'similarity': similarities[top_idx]
    }