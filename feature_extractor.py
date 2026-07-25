# ==========================================================
# feature_extractor.py - Feature Extraction using DeepFace
# ==========================================================

import os
import pickle
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
        # Step 1: Read image using PIL
        try:
            img_pil = Image.open(img_path)
            img = np.array(img_pil)
        except Exception as e:
            print(f"⚠️ Failed to read image {img_path}: {e}")
            return None

        # Step 2: Detect faces using MTCNN (expects RGB)
        try:
            results = self.detector.detect_faces(img)
        except Exception as e:
            print(f"⚠️ Face detection failed for {img_path}: {e}")
            return None

        if not results:
            return None

        # Step 3: Extract the first detected face
        try:
            # Get bounding box coordinates
            x, y, width, height = results[0]['box']
            x, y = max(0, x), max(0, y)
            width, height = abs(width), abs(height)

            # Crop the face from the original image
            face_array = img[y:y + height, x:x + width]

            if face_array.size == 0:
                return None

            # Convert numpy array to PIL Image for saving
            face_pil = Image.fromarray(face_array)

            # Step 4: Save face temporarily for DeepFace processing
            temp_face_path = "temp_face.jpg"
            face_pil.save(temp_face_path)

            # Step 5: Extract features using DeepFace
            try:
                list_of_dicts = DeepFace.represent(
                    img_path=temp_face_path,
                    model_name=self.model_name,
                    detector_backend='skip',
                    enforce_detection=False
                )

                # Clean up temporary file
                if os.path.exists(temp_face_path):
                    os.remove(temp_face_path)

                if list_of_dicts and len(list_of_dicts) > 0:
                    first_face_dict = list_of_dicts[0]
                    feature_vector = np.array(first_face_dict['embedding'])
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

    if not os.path.exists('data'):
        print("❌ 'data' folder not found! Please create it and add celebrity images.")
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

    # Save filenames
    os.makedirs('models', exist_ok=True)
    with open('models/filenames.pkl', 'wb') as f:
        pickle.dump(filenames, f)
    print("💾 Saved models/filenames.pkl")

    print("\n🔄 Initializing feature extractor...")
    extractor = FeatureExtractor(model_name="VGG-Face")

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

    with open('models/embeddings.pkl', 'wb') as f:
        pickle.dump(np.array(feature_vectors), f)
    print("💾 Saved models/embeddings.pkl")

    print(f"\n✅ Feature extraction complete!")
    print(f"📊 Successfully processed: {len(feature_vectors)} images")
    print(f"⚠️ Skipped: {failed_count} images (no face or error)")
    print("=" * 60)


if __name__ == "__main__":
    batch_extract_features()