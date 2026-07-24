#!/bin/bash
# setup.sh - Complete project setup

echo "=========================================="
echo "🎬 Bollywood Celebrity Detector Setup"
echo "=========================================="

# Create directories
echo "📁 Creating directories..."
mkdir -p data models uploads
mkdir -p data/actor1 data/actor2  # Example structure

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt


echo "=========================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Extract features: python feature_extractor.py"
echo "2. Run the app: streamlit run app.py"
echo "=========================================="