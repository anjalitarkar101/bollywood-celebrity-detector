# 🎬 Bollywood Celebrity Detector

## 📖 Overview
A **Bollywood Celebrity Detector** built with Streamlit and Deep Learning that finds your celebrity lookalike. The system uses MTCNN for face detection and DeepFace (VGG-Face) for facial feature extraction to match your uploaded photo with a database of 100 Bollywood celebrities.

---

## ✨ Features
- 🎭 Celebrity Detection - Finds your Bollywood celebrity lookalike using Deep Learning
- 📸 Face Detection - Uses MTCNN for accurate face detection
- 🔍 Feature Extraction - Utilizes DeepFace (VGG-Face) for facial feature extraction
- 📊 Confidence Score - Shows similarity confidence percentage
- 🎯 Real-time Results - Instant matching with visual feedback
- 🎨 Clean UI - User-friendly interface with side-by-side comparison

---

## 🛠️ Technologies Used
- Python 3.10+ - Core programming language
- DeepFace - Facial recognition and feature extraction
- MTCNN - Face detection
- Streamlit - Web application framework
- OpenCV - Image processing
- NumPy - Numerical operations
- Scikit-learn - Similarity calculation (Cosine Similarity)

---

## 📁 Project Structure
```text
bollywood-celebrity-detector/
├── app.py                    # Main Streamlit application (UI)
├── predict.py                # Prediction functions
├── feature_extractor.py      # Feature extraction script
├── requirements.txt          # Python dependencies
├── setup.sh                  # Setup script
├── .gitignore               # Git ignore file
├── data/                     # Celebrity images (gitignored)
│   ├── Actor_Name1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── Actor_Name2/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── ...
├── models/                   # Generated features (gitignored)
│   ├── embeddings.pkl
│   └── filenames.pkl
├── uploads/                  # Temporary uploads (gitignored)
└── README.md                 # Project documentation
```


---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/anjalitarkar101/bollywood-celebrity-detector.git
cd bollywood-celebrity-detector
```

### Step 2: Run Setup Script
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create required directories (data/, models/, uploads/)
- Install all dependencies


### Step 3: Download Dataset

Download the dataset from Kaggle:

- **Source:** Kaggle
- **Link:** https://www.kaggle.com/datasets/sushilyadav1998/bollywood-celeb-localized-face-dataset
- **Name:** Bollywood Celeb Localized Face Dataset

After downloading, extract the files and organize them as follows:

```text
data/
├── Actor_Name1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Actor_Name2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

```text
Replace `Actor_Name1`, `Actor_Name2`, etc., with the actual names of the Bollywood celebrities from the dataset.
```

### Step 4: Extract Features
```bash
python feature_extractor.py
```
This will:
- Load all celebrity images
- Detect faces using MTCNN
- Extract features using DeepFace (VGG-Face)
- Save embeddings and filenames to models/ folder

### Step 5: Run the Application
```bash
streamlit run app.py
Open your browser and navigate to http://localhost:8501
```

---

## 📊 How It Works

1. Face Detection (MTCNN)
- Detects faces in the uploaded image
- Crops the face region
- Handles multiple faces (uses the first detected face)

2. Feature Extraction (DeepFace - VGG-Face)
- Extracts 4096-dimensional feature vectors
- Normalizes features using L2 normalization
- Creates embeddings for each celebrity image

3. Similarity Calculation (Cosine Similarity)
- Compares uploaded face features with celebrity database
- Calculates cosine similarity scores
- Returns the best match with confidence percentage

4. Prediction Pipeline
```txt
Upload Image
    ↓
Face Detection (MTCNN)
    ↓
Face Crop
    ↓
Feature Extraction (DeepFace - VGG-Face)
    ↓
Cosine Similarity with Celebrity Database
    ↓
Best Match with Confidence Score
```

---

## 🔧 Dependencies
```txt
tensorflow-macos==2.13.0
tensorflow-metal==1.2.0
mtcnn==0.1.0
opencv-python==4.8.1.78
streamlit==1.50.0
numpy==1.24.3
scikit-learn==1.3.0
Pillow==10.0.0
tqdm==4.66.0
deepface==0.0.79
```

---

## 📊 Dataset Information

| Feature | Details |
|---------|---------|
| Celebrities | 100 Bollywood celebrities |
| Images per celebrity | 80-150 samples |
| Image Size | 64x64 pixels |
| Conditions | Wild (different orientations, illuminations, age transitions) |


---

## 📝 Usage Guide
1. Upload Photo - Click "Choose an image" and upload a clear front-facing photo
2. Face Detection - AI detects and extracts your face
3. Feature Extraction - Face features are extracted using DeepFace
4. Celebrity Match - System finds the best celebrity match
5. View Results - See your celebrity lookalike with confidence score


---

## 📸 Tips for Best Results
- ✅ Use a clear front-facing photo
- 💡 Ensure good lighting
- 👀 Make sure your face is visible
- 📸 Avoid blurry or low-quality images
- 🎯 Try different angles for best results


---

## 📄 License
This project is licensed under the MIT License.

© 2026 Anjali Tarkar. All rights reserved.

---

## 👩‍💻 Author
**Anjali Tarkar**
- GitHub: https://github.com/anjalitarkar101
- Email: anjalitarkar101@gmail.com

---

## ⭐ Show Your Support
If you find this project useful, please give it a star on GitHub!

---

## 🙏 Acknowledgments
- DeepFace - For facial recognition and feature extraction
- MTCNN - For face detection
- Streamlit - For the awesome web framework
- Sushil Kumar Yadav - For the Bollywood Celeb Localized Face Dataset

