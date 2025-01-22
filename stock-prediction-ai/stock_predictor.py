import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

class StockPredictor:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()

    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        stock = yf.Ticker(self.symbol)
        self.data = stock.history(start=self.start_date, end=self.end_date)
        return self.data

    def prepare_data(self, sequence_length=60):
        """Prepare data for LSTM model"""
        # Use closing prices
        dataset = self.data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(dataset)
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(dataset)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test sets
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        # Reshape for LSTM [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        return X_train, y_train, X_test, y_test

    def build_model(self, sequence_length):
        """Build LSTM model"""
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        return self.model

    def train_model(self, X_train, y_train, epochs=50, batch_size=32):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        return history

    def predict(self, X_test):
        """Make predictions"""
        predictions = self.model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions)
        return predictions

    def plot_predictions(self, actual_values, predicted_values):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(15,6))
        plt.plot(actual_values, label='Actual')
        plt.plot(predicted_values, label='Predicted')
        plt.title(f'{self.symbol} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

def main():
    # Initialize predictor
    predictor = StockPredictor(
        symbol='AAPL',
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # Fetch data
    data = predictor.fetch_data()
    print("Data fetched successfully!")
    
    # Prepare data
    X_train, y_train, X_test, y_test = predictor.prepare_data()
    print("Data prepared for training!")
    
    # Build and train model
    model = predictor.build_model(sequence_length=60)
    history = predictor.train_model(X_train, y_train)
    print("Model training completed!")
    
    # Make predictions
    predictions = predictor.predict(X_test)
    
    # Plot results
    actual_values = predictor.data['Close'].values[-(len(predictions)):]
    predictor.plot_predictions(actual_values, predictions)

if __name__ == "__main__":
    main()

