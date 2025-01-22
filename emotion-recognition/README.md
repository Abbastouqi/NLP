# Emotion Recognition using Deep Learning

This project implements real-time emotion recognition using Convolutional Neural Networks (CNN) and OpenCV. It can detect seven different emotions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Features
- Real-time facial emotion detection
- CNN-based deep learning model
- Webcam integration
- Data augmentation support
- Training capability with custom datasets

## Setup
1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate virtual environment
4. Install requirements: `pip install -r requirements.txt`

## Usage
Run the emotion recognition system:
```python emotion_recognition.py```

## Model Architecture
- Multiple Conv2D layers
- MaxPooling layers
- Dropout for regularization
- Dense layers for classification
- Softmax activation for multi-class output