# Advanced AI/ML Projects Collection

A comprehensive collection of AI and Machine Learning projects showcasing various applications from NLP to Computer Vision.

## Projects Overview

### 1. Keyword Generation with Transformer-Based Models
Advanced NLP implementation using BERT embeddings and GPT-2 with LoRA layers for efficient keyword generation.

### 2. LLM Applications
Practical applications of Large Language Models including content summarization, text classification, and sentiment analysis.

### 3. Stock Market Prediction using AI
LSTM-based neural network implementation for stock market price prediction.

### 4. Emotion Recognition using Deep Learning
Real-time emotion recognition using CNN and OpenCV.

## Detailed Project Descriptions

### 1. Keyword Generation Project



#### Features
- BERT embeddings for text representation
- GPT-2 with LoRA layers for efficient fine-tuning
- Custom dataset handling for marketing data
- LoRA Configuration:
  - r=8 (rank dimension)
  - lora_alpha=32
  - Targeted attention modules
  - 0.1 dropout rate

### 2. LLM Applications
#### Features
- Content Summarization
- Text Classification
- Sentiment Analysis
- Zero-shot Classification with custom labels

### 3. Stock Market Prediction
#### Features
- Real-time stock data fetching
- LSTM neural network architecture
- Data preprocessing and scaling
- Visual prediction results
- Historical price analysis

### 4. Emotion Recognition
#### Features
- Real-time facial emotion detection
- Seven emotion classifications
- CNN-based architecture
- Webcam integration
- Data augmentation support

## Installation & Setup

1. Clone the repository
```bash
git clone https://github.com/Abbastouqi/NLP.git
cd NLP






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
