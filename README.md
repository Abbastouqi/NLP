# AI/ML Projects Collection

## Project 1: LLM Applications
This project demonstrates three practical applications of Large Language Models:
1. Content Summarization
2. Text Classification
3. Sentiment Analysis

### Setup
1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate virtual environment: `.\venv\Scripts\activate` (Windows)
4. Install requirements: `pip install -r requirements.txt`

### Features
- Text Summarization using BART model
- Zero-shot Classification with custom labels
- Sentiment Analysis
---

## Project 2: Stock Market Prediction using AI

This project uses LSTM (Long Short-Term Memory) neural networks to predict stock market prices. It features:
- Real-time stock data fetching using Yahoo Finance
- Data preprocessing and scaling
- LSTM-based deep learning model
- Visual prediction results
- Historical price analysis

### Features
- Fetches real stock market data
- Uses LSTM neural network for time series prediction
- Includes data visualization
- Supports any stock symbol available on Yahoo Finance
- Configurable prediction parameters

### Model Architecture
- 3 LSTM layers with dropout
- Dense output layer
- Adam optimizer
- Mean Squared Error loss function
---

## Project 3: Emotion Recognition using Deep Learning

This project implements real-time emotion recognition using Convolutional Neural Networks (CNN) and OpenCV. It can detect seven different emotions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

### Features
- Real-time facial emotion detection
- CNN-based deep learning model
- Webcam integration
- Data augmentation support
- Training capability with custom datasets

### Model Architecture
- Multiple Conv2D layers
- MaxPooling layers
- Dropout for regularization
- Dense layers for classification
- Softmax activation for multi-class output

## Common Setup for All Projects
1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate virtual environment
4. Install requirements: `pip install -r requirements.txt`

## Technologies Used
- TensorFlow/Keras
- PyTorch
- Hugging Face Transformers
- OpenCV
- NumPy/Pandas
- Scikit-learn
- Matplotlib

## License
MIT License