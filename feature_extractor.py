# ==========================================================
# feature_extractor.py - Feature Extraction using DeepFace
# ==========================================================

import os
import pickle
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from mtcnn import MTCNN
from deepface import DeepFace


class FeatureExtractor:

    def __init__(self, model_name="VGG-Face"):
        print("🔄 Loading face detector model ...")
        self.detector = MTCNN()
        print("✅ Face detector ready!")

        print(f"🔄 Loading DeepFace with {model_name} model...")
        self.model_name = model_name
        print(f"✅ Feature extractor ready!")

    def extract_features(self, img_path):
        # Step 1: Read image from disk in BGR format
        img = cv2.imread(img_path)
        if img is None:
            return None

        # Step 2: Detect faces using MTCNN which expects RGB format
        try:
            results = self.detector.detect_faces(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            )
        except Exception as e:
            print(f"⚠️ Face detection failed for {img_path}: {e}")
            return None

        if not results:
            return None

        # Step 3: Extract the first detected face
        try:
            # Get bounding box coordinates of the first face
            x, y, width, height = results[0]['box']

            # Ensure coordinates are valid
            x, y = max(0, x), max(0, y)
            width, height = abs(width), abs(height)

            # Crop the face from the original image
            face = img[y:y + height, x:x + width]

            # Check if face is empty
            if face.size == 0 or face is None:
                return None

            # Step 4: Save face temporarily for DeepFace processing
            temp_face_path = "temp_face.jpg"
            cv2.imwrite(temp_face_path, face)

            # Step 5: Extract features using DeepFace
            try:
                list_of_dicts = DeepFace.represent(
                    img_path=temp_face_path,
                    model_name=self.model_name,
                    detector_backend='skip',  # to skip detection
                    enforce_detection=False   # to avoid errors if face is not detected
                )

                # Clean up temporary file
                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)

                # Check if we got valid data
                if list_of_dicts and len(list_of_dicts) > 0:
                    # Get the first face's data (the dictionary)
                    first_face_dict = list_of_dicts[0]

                    # Extract the feature vector from the dictionary
                    feature_vector = np.array(first_face_dict['embedding'])

                    # Normalize feature vector (L2 normalization)
                    normalized_feature_vector = feature_vector / np.linalg.norm(feature_vector)
                    return normalized_feature_vector

                return None

            except Exception as e:
                print(f"⚠️ DeepFace feature extraction failed: {e}")
                return None

        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")
            return None


def batch_extract_features():
    print("=" * 60)
    print("🎬 Bollywood Celebrity Feature Extractor (DeepFace)")
    print("=" * 60)

    # Check if data folder exists
    if not os.path.exists('data'):
        print("❌ 'data' folder not found! Please create it and add celebrity images.")
        return

    # Get all actor folders
    actors = [d for d in os.listdir('data') if os.path.isdir(os.path.join('data', d))]

    if not actors:
        print("❌ No actor folders found in 'data'!")
        return
    print(f"📁 Found {len(actors)} actors")

    # Get all image paths
    print("\n🔄 Getting paths of all images ...")
    filenames = []
    for actor in actors:
        actor_path = os.path.join('data', actor)
        for file in os.listdir(actor_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_path = os.path.join('data', actor, file)
                # Check if file is not empty (at least 1KB)
                if os.path.getsize(file_path) > 1000:
                    filenames.append(file_path)
    print(f"🖼️  Found {len(filenames)} valid images")

    # Save filenames
    with open(' models/filenames.pkl', 'wb') as f:
        pickle.dump(filenames, f)
    print("💾 Saved models/filenames.pkl")

    # Initialize feature extractor
    print("\n🔄 Initializing feature extractor...")
    extractor = FeatureExtractor(model_name="VGG-Face")

    # Extract features for all images
    print("\n🔄 Extracting features...")
    print(f"⚠️ Images with no faces will be skipped")
    feature_vectors = []
    failed_count = 0

    for file in tqdm(filenames, desc="Processing images"):
        feature_vector = extractor.extract_features(file)
        if feature_vector is not None:
            feature_vectors.append(feature_vector)
        else:
            failed_count += 1

    # Save features
    with open('models/embeddings.pkl', 'wb') as f:
        pickle.dump(np.array(feature_vectors), f)
    print("💾 Saved models/embeddings.pkl")

    print(f"\n✅ Feature extraction complete!")
    print(f"📊 Successfully processed: {len(feature_vectors)} images")
    print(f"⚠️ Skipped: {failed_count} images (no face or error)")
    print("=" * 60)


if __name__ == "__main__":
    batch_extract_features()