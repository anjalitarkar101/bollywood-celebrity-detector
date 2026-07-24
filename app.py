# ==========================================
# app.py - Bollywood Celebrity Detector
# ==========================================

import os
import streamlit as st
from PIL import Image
from predict import get_extractor, load_data, find_celebrity_match


# ==========================================
# Main App
# ==========================================
def main():
    # Page Configuration
    st.set_page_config(
        page_title="Bollywood Celebrity Detector Using CNN",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ==========================================
    # Sidebar
    # ==========================================
    with st.sidebar:
        st.markdown("### 🎭 About This App")
        st.markdown("""
            This app uses **Deep Learning** to find your celebrity lookalike!

            **How it works:**
            1. 📤 Upload your photo
            2. 🔍 AI detects your face
            3. 🎯 Finds your celebrity match
            4. 📊 Shows confidence score
        """)

        st.markdown("---")

        st.markdown("### 🧠 Model Info")
        st.markdown("""
            - **Face Detection:** MTCNN
            - **Feature Extraction:** DeepFace (VGG-Face)
            - **Similarity:** Cosine Similarity
            - **Celebrities:** Bollywood Actors
        """)

        st.markdown("---")

        st.markdown("### 📸 Tips")
        st.markdown("""
            - ✅ Use a clear front-facing photo
            - 💡 Ensure good lighting
            - 👀 Make sure your face is visible
            - 📸 Avoid blurry or low-quality images
            - 🎯 Try different angles for best results
        """)

    # ==========================================
    # Main Content
    # ==========================================

    st.title("🎬 Which Bollywood Celebrity Are You?")
    st.markdown("Upload your photo and find your celebrity lookalike!")

    # Load models
    with st.spinner("🔄 Loading face detector and feature extractor..."):
        extractor = get_extractor()

    # Load data
    with st.spinner("🔄 Loading celebrity database..."):
        feature_vectors, filenames = load_data()

    # Check if data loaded
    if feature_vectors is None or filenames is None:
        st.error("❌ Feature files not found! Please run: python feature_extractor.py")
        st.stop()

    st.success("✅ Model and data loaded successfully!")

    # ==========================================
    # File Upload Section
    # ==========================================
    st.markdown("---")

    uploaded_image = st.file_uploader(
        "📤 Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear front-facing photo"
    )

    # ==========================================
    # Process Uploaded Image
    # ==========================================
    if uploaded_image is not None:
        # Save uploaded image
        file_path = os.path.join('uploads', uploaded_image.name)

        try:
            with open(file_path, 'wb') as f:
                f.write(uploaded_image.getbuffer())

            # Continue with processing
            col1, col2 = st.columns(2)

            # ========== Column 1: Your Photo ==========
            with col1:
                st.subheader("📸 Your Photo")

                display_image = Image.open(uploaded_image)
                st.image(display_image, width=300)

                # Show image info
                st.caption(f"📄 {uploaded_image.name}")
                st.caption(f"📦 {uploaded_image.size / 1024:.1f} KB")
                st.caption(f"📐 {display_image.width} x {display_image.height} px")

            # ========== Column 2: Prediction ==========
            with col2:
                st.subheader("🔍 Finding Your Match...")

                # Extract features from uploaded image
                with st.spinner("🔍 Detecting face and extracting features..."):
                    uploaded_feature_vector = extractor.extract_features(
                        os.path.join('uploads', uploaded_image.name)
                    )

                if uploaded_feature_vector is None:
                    st.error("❌ No face detected! Please upload a clear photo with a visible face.")
                    st.stop()

                # Find the best celebrity match
                top_match = find_celebrity_match(
                    feature_vectors,
                    uploaded_feature_vector,
                    filenames
                )

                # ========== Display Result ==========
                st.markdown(f"### 🎭 **{top_match['name']}**")
                st.image(top_match['image_path'], width=300)

                # Show confidence score
                confidence = top_match['similarity'] * 100
                st.progress(min(confidence / 100, 1.0))

                if confidence >= 60:
                    st.success(f"✅ Match Confidence: {confidence:.1f}%")
                elif confidence >= 40:
                    st.warning(f"⚠️ Match Confidence: {confidence:.1f}%")
                else:
                    st.error(f"❌ Match Confidence: {confidence:.1f}%")

        except Exception as e:
            st.error(f"Error saving file: {e}")

    # ==========================================
    # Footer
    # ==========================================
    st.markdown("---")
    st.caption("🔍 Powered by MTCNN (Face Detection) + DeepFace (Feature Extraction)")


if __name__ == "__main__":
    main()