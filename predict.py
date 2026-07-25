# ==========================================================
# predict.py - Bollywood Celebrity Detector
# ==========================================================

import os
import pickle
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace


class FeatureExtractor:
    def __init__(self, model_name="VGG-Face"):
        self.model_name = model_name

    def extract_features(self, img_path):
        try:
            result = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                detector_backend='opencv',
                enforce_detection=False
            )

            if result and len(result) > 0:
                feature_vector = np.array(result[0]['embedding'])
                return feature_vector / np.linalg.norm(feature_vector)

            return None
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None


@st.cache_resource
def get_extractor():
    return FeatureExtractor(model_name="VGG-Face")


@st.cache_data
def load_data():
    try:
        with open('models/embeddings.pkl', 'rb') as f:
            feature_vectors = pickle.load(f)

        with open('models/filenames.pkl', 'rb') as f:
            filenames = pickle.load(f)

        return feature_vectors, filenames
    except FileNotFoundError:
        return None, None


def find_celebrity_match(feature_vectors, uploaded_features, filenames):
    similarities = cosine_similarity([uploaded_features], feature_vectors)[0]
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]

    file_path = filenames[best_idx]
    celebrity_name = file_path.split(os.sep)[1] if os.sep in file_path else file_path.split('/')[1]

    return {
        'name': celebrity_name,
        'image_path': file_path,
        'similarity': best_similarity
    }