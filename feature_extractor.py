# ==========================================================
# feature_extractor.py - Feature Extraction using DeepFace
# ==========================================================

import os
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from deepface import DeepFace


class FeatureExtractor:

    def __init__(self, model_name="VGG-Face"):
        print(f"🔄 Loading DeepFace with {model_name} model...")
        self.model_name = model_name
        print(f"✅ Feature extractor ready!")

    def extract_features(self, img_path):
        """Extract features using DeepFace (with built-in detection)"""
        try:
            list_of_dicts = DeepFace.represent(
                img_path=img_path,
                model_name=self.model_name,
                detector_backend='opencv',
                enforce_detection=False
            )

            if list_of_dicts and len(list_of_dicts) > 0:
                first_face_dict = list_of_dicts[0]
                feature_vector = np.array(first_face_dict['embedding'])
                normalized_feature_vector = feature_vector / np.linalg.norm(feature_vector)
                return normalized_feature_vector

            return None

        except Exception as e:
            print(f"⚠️ Feature extraction failed for {img_path}: {e}")
            return None


def batch_extract_features():
    print("=" * 60)
    print("🎬 Bollywood Celebrity Feature Extractor (DeepFace)")
    print("=" * 60)

    if not os.path.exists('data'):
        print("❌ 'data' folder not found!")
        return

    actors = [d for d in os.listdir('data') if os.path.isdir(os.path.join('data', d))]

    if not actors:
        print("❌ No actor folders found in 'data'!")
        return
    print(f"📁 Found {len(actors)} actors")

    print("\n🔄 Getting paths of all images ...")
    filenames = []
    for actor in actors:
        actor_path = os.path.join('data', actor)
        for file in os.listdir(actor_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_path = os.path.join('data', actor, file)
                if os.path.getsize(file_path) > 1000:
                    filenames.append(file_path)
    print(f"🖼️ Found {len(filenames)} valid images")

    os.makedirs('models', exist_ok=True)
    with open('models/filenames.pkl', 'wb') as f:
        pickle.dump(filenames, f)
    print("💾 Saved models/filenames.pkl")

    print("\n🔄 Initializing feature extractor...")
    extractor = FeatureExtractor(model_name="VGG-Face")

    print("\n🔄 Extracting features...")
    feature_vectors = []
    failed_count = 0

    for file in tqdm(filenames, desc="Processing images"):
        feature_vector = extractor.extract_features(file)
        if feature_vector is not None:
            feature_vectors.append(feature_vector)
        else:
            failed_count += 1

    with open('models/embeddings.pkl', 'wb') as f:
        pickle.dump(np.array(feature_vectors), f)
    print("💾 Saved models/embeddings.pkl")

    print(f"\n✅ Feature extraction complete!")
    print(f"📊 Successfully processed: {len(feature_vectors)} images")
    print(f"⚠️ Skipped: {failed_count} images (no face or error)")
    print("=" * 60)


if __name__ == "__main__":
    batch_extract_features()