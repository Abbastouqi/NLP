# Stock Market Prediction using AI

This project uses LSTM (Long Short-Term Memory) neural networks to predict stock market prices. It features:

- Real-time stock data fetching using Yahoo Finance
- Data preprocessing and scaling
- LSTM-based deep learning model
- Visual prediction results
- Historical price analysis

## Features
- Fetches real stock market data
- Uses LSTM neural network for time series prediction
- Includes data visualization
- Supports any stock symbol available on Yahoo Finance
- Configurable prediction parameters

## Setup
1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate virtual environment
4. Install requirements: `pip install -r requirements.txt`

## Usage
Run the predictor:
```python stock_predictor.py```

## Model Architecture
- 3 LSTM layers with dropout
- Dense output layer
- Adam optimizer
- Mean Squared Error loss function