"""
Market Predictor Module.

This module combines different prediction approaches to forecast stock movements,
including machine learning models and technical analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketPredictor:
    """
    Market predictor for stock price forecasting.
    
    This class combines different prediction models, including machine learning
    and technical analysis, to forecast stock price movements.
    """

    def __init__(self, use_deep_learning: bool = True, use_ml: bool = True):
        """
        Initialize the market predictor.

        Args:
            use_deep_learning: Whether to use deep learning models
            use_ml: Whether to use traditional machine learning models
        """
        self.use_deep_learning = use_deep_learning
        self.use_ml = use_ml
        self.dl_model = None
        self.ml_model = None
        self.scalers = {}
        self.feature_columns = []
        
        logger.info(f"MarketPredictor initialized (DL: {use_deep_learning}, ML: {use_ml})")

    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build an LSTM model for time series prediction.

        Args:
            input_shape: Shape of the input data (time_steps, features)

        Returns:
            Compiled Keras model
        """
        try:
            model = Sequential()
            
            # Add LSTM layers
            model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            
            model.add(LSTM(units=50, return_sequences=False))
            model.add(Dropout(0.2))
            
            # Output layer
            model.add(Dense(units=1))
            
            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            logger.info(f"LSTM model built with input shape {input_shape}")
            return model
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {e}")
            return None

    def prepare_data(self, X: np.ndarray, y: np.ndarray, time_steps: int = 10) -> Tuple:
        """
        Prepare data for LSTM model by creating sequences.

        Args:
            X: Feature matrix
            y: Target vector
            time_steps: Number of time steps to include in each sequence

        Returns:
            Tuple of (X_lstm, y_lstm, scaler_x, scaler_y)
        """
        try:
            # Scale the data
            scaler_x = MinMaxScaler(feature_range=(0, 1))
            scaler_y = MinMaxScaler(feature_range=(0, 1))
            
            X_scaled = scaler_x.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
            
            # Create sequences
            X_lstm, y_lstm = [], []
            
            for i in range(len(X_scaled) - time_steps):
                X_lstm.append(X_scaled[i:i+time_steps])
                y_lstm.append(y_scaled[i+time_steps])
                
            X_lstm = np.array(X_lstm)
            y_lstm = np.array(y_lstm)
            
            logger.info(f"Prepared LSTM data with shape X: {X_lstm.shape}, y: {y_lstm.shape}")
            return X_lstm, y_lstm, scaler_x, scaler_y
            
        except Exception as e:
            logger.error(f"Error preparing data for LSTM: {e}")
            return None, None, None, None

    def train_lstm_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        time_steps: int = 10,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> bool:
        """
        Train an LSTM model for time series prediction.

        Args:
            X: Feature matrix
            y: Target vector
            time_steps: Number of time steps for sequences
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Portion of data to use for validation

        Returns:
            Whether training was successful
        """
        if not self.use_deep_learning:
            logger.info("Deep learning is disabled")
            return False
            
        try:
            # Prepare data
            X_lstm, y_lstm, scaler_x, scaler_y = self.prepare_data(X, y, time_steps)
            
            if X_lstm is None:
                return False
                
            # Save scalers
            self.scalers['X'] = scaler_x
            self.scalers['y'] = scaler_y
            
            # Build model
            input_shape = (X_lstm.shape[1], X_lstm.shape[2])
            self.dl_model = self.build_lstm_model(input_shape)
            
            if self.dl_model is None:
                return False
                
            # Train model
            self.dl_model.fit(
                X_lstm, y_lstm,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1
            )
            
            logger.info("LSTM model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return False

    def train_ml_model(self, X: np.ndarray, y: np.ndarray) -> bool:
        """
        Train a machine learning model for prediction.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Whether training was successful
        """
        if not self.use_ml:
            logger.info("Machine learning is disabled")
            return False
            
        try:
            # Initialize and train a Random Forest model
            self.ml_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.ml_model.fit(X, y)
            
            # Store feature columns if available
            if hasattr(X, 'columns'):
                self.feature_columns = list(X.columns)
                
            logger.info("Random Forest model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            return False

    def predict_with_lstm(self, X: np.ndarray, time_steps: int = 10) -> np.ndarray:
        """
        Make predictions using the LSTM model.

        Args:
            X: Feature matrix
            time_steps: Number of time steps for sequences

        Returns:
            Array of predictions
        """
        if not self.use_deep_learning or self.dl_model is None:
            logger.warning("LSTM model not available")
            return None
            
        try:
            # Scale the input
            X_scaled = self.scalers['X'].transform(X)
            
            # Create sequences
            X_lstm = []
            for i in range(len(X_scaled) - time_steps + 1):
                X_lstm.append(X_scaled[i:i+time_steps])
                
            X_lstm = np.array(X_lstm)
            
            # Make predictions
            y_pred_scaled = self.dl_model.predict(X_lstm)
            
            # Inverse transform to get original scale
            y_pred = self.scalers['y'].inverse_transform(y_pred_scaled)
            
            return y_pred.flatten()
            
        except Exception as e:
            logger.error(f"Error predicting with LSTM: {e}")
            return None

    def predict_with_ml(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the machine learning model.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        if not self.use_ml or self.ml_model is None:
            logger.warning("ML model not available")
            return None
            
        try:
            # Make predictions
            y_pred = self.ml_model.predict(X)
            
            return y_pred
            
        except Exception as e:
            logger.error(f"Error predicting with ML model: {e}")
            return None

    def ensemble_predict(self, X: np.ndarray, time_steps: int = 10) -> np.ndarray:
        """
        Make predictions using an ensemble of models.

        Args:
            X: Feature matrix
            time_steps: Number of time steps for LSTM

        Returns:
            Array of ensemble predictions
        """
        predictions = []
        weights = []
        
        # Get LSTM predictions if available
        if self.use_deep_learning and self.dl_model is not None:
            lstm_preds = self.predict_with_lstm(X, time_steps)
            if lstm_preds is not None:
                predictions.append(lstm_preds)
                weights.append(0.6)  # Higher weight for deep learning
        
        # Get ML predictions if available
        if self.use_ml and self.ml_model is not None:
            ml_preds = self.predict_with_ml(X)
            if ml_preds is not None:
                predictions.append(ml_preds)
                weights.append(0.4)  # Lower weight for traditional ML
        
        if not predictions:
            logger.warning("No predictions available from any model")
            return None
            
        # If we only have one model's predictions, return them
        if len(predictions) == 1:
            return predictions[0]
            
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Ensemble the predictions using weighted average
        ensemble_preds = np.zeros_like(predictions[0])
        for i, preds in enumerate(predictions):
            ensemble_preds += weights[i] * preds
            
        return ensemble_preds

    def train_for_ticker(
        self, 
        df: pd.DataFrame, 
        ticker: str, 
        target_column: str = 'close',
        forecast_horizon: int = 5
    ) -> Dict:
        """
        Train models for a specific ticker.

        Args:
            df: DataFrame with historical data
            ticker: Stock symbol
            target_column: Column to predict
            forecast_horizon: Number of days to forecast

        Returns:
            Dictionary with training results
        """
        try:
            logger.info(f"Training models for {ticker}")
            
            # Make sure DataFrame is not empty
            if df.empty:
                return {'error': f'No data available for {ticker}'}
                
            # Create target variable (next day's price)
            df['target'] = df[target_column].shift(-forecast_horizon)
            
            # Drop NaN values
            df = df.dropna()
            
            # Select features
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'return_1d', 'return_5d',
                'rolling_mean_5d', 'rolling_mean_20d',
                'daily_volatility', 'weekly_volatility',
                'volume_change_1d',
                'sma_20', 'sma_50', 
                'rsi_14', 'macd', 'macd_signal', 
                'bollinger_width', 'atr'
            ]
            
            # Add sentiment features if they exist
            sentiment_columns = [col for col in df.columns if 'sentiment' in col.lower()]
            feature_columns.extend(sentiment_columns)
            
            # Select only columns that exist in the DataFrame
            feature_columns = [col for col in feature_columns if col in df.columns]
            
            # Prepare data
            X = df[feature_columns].values
            y = df['target'].values
            
            # Train models
            lstm_success = False
            ml_success = False
            
            if self.use_deep_learning:
                lstm_success = self.train_lstm_model(
                    X, y,
                    time_steps=10,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2
                )
                
            if self.use_ml:
                ml_success = self.train_ml_model(X, y)
                
            return {
                'ticker': ticker,
                'feature_columns': feature_columns,
                'lstm_trained': lstm_success,
                'ml_trained': ml_success,
                'data_shape': X.shape,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training models for {ticker}: {e}")
            return {'error': str(e)}

    def predict_for_ticker(
        self, 
        df: pd.DataFrame, 
        ticker: str, 
        days_to_predict: int = 1
    ) -> Dict:
        """
        Make predictions for a specific ticker.

        Args:
            df: DataFrame with historical data
            ticker: Stock symbol
            days_to_predict: Number of days to predict ahead

        Returns:
            Dictionary with prediction results
        """
        try:
            logger.info(f"Making predictions for {ticker}")
            
            # Make sure DataFrame is not empty
            if df.empty:
                return {'error': f'No data available for {ticker}'}
                
            # Select features (use the same as during training)
            if not self.feature_columns:
                # Default feature columns if not set during training
                feature_columns = [
                    'open', 'high', 'low', 'close', 'volume',
                    'return_1d', 'return_5d',
                    'rolling_mean_5d', 'rolling_mean_20d',
                    'daily_volatility', 'volume_change_1d',
                    'sma_20', 'sma_50', 'rsi_14'
                ]
                # Select only columns that exist in the DataFrame
                feature_columns = [col for col in feature_columns if col in df.columns]
            else:
                feature_columns = self.feature_columns
                
            # Prepare data
            X = df[feature_columns].values
            
            # Get last price
            last_price = df['close'].iloc[-1]
            
            # Make predictions
            predictions = self.ensemble_predict(X, time_steps=10)
            
            if predictions is None:
                return {
                    'ticker': ticker,
                    'error': 'No predictions available'
                }
                
            # Get prediction for the specified days ahead
            # For simplicity, we'll just use the last prediction
            predicted_price = predictions[-1]
            
            # Calculate percentage change
            percent_change = (predicted_price - last_price) / last_price * 100
            
            return {
                'ticker': ticker,
                'last_price': last_price,
                'predicted_price': predicted_price,
                'percent_change': percent_change,
                'days_ahead': days_to_predict,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making predictions for {ticker}: {e}")
            return {'error': str(e)}

    def get_model_importance(self) -> Dict:
        """
        Get feature importance from the ML model.

        Returns:
            Dictionary with feature importance
        """
        if not self.use_ml or self.ml_model is None:
            logger.warning("ML model not available for feature importance")
            return {'error': 'ML model not available'}
            
        try:
            # Get feature importance from Random Forest
            importance = self.ml_model.feature_importances_
            
            # Create a dictionary mapping features to importance
            if not self.feature_columns:
                feature_names = [f"feature_{i}" for i in range(len(importance))]
            else:
                feature_names = self.feature_columns
                
            importance_dict = {}
            for i, feature in enumerate(feature_names):
                if i < len(importance):
                    importance_dict[feature] = float(importance[i])
                    
            # Sort by importance (descending)
            importance_dict = {k: v for k, v in sorted(
                importance_dict.items(), 
                key=lambda item: item[1], 
                reverse=True
            )}
            
            return {
                'feature_importance': importance_dict,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting model importance: {e}")
            return {'error': str(e)}

    def save_model(self, directory: str, ticker: str) -> bool:
        """
        Save the trained models.

        Args:
            directory: Directory to save models
            ticker: Stock symbol for filename

        Returns:
            Whether saving was successful
        """
        try:
            import os
            import pickle
            from pathlib import Path
            
            # Create directory if it doesn't exist
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            # Save LSTM model if available
            if self.use_deep_learning and self.dl_model is not None:
                lstm_path = os.path.join(directory, f"{ticker}_lstm_model.h5")
                self.dl_model.save(lstm_path)
                logger.info(f"LSTM model saved to {lstm_path}")
                
                # Save scalers
                scalers_path = os.path.join(directory, f"{ticker}_scalers.pkl")
                with open(scalers_path, 'wb') as f:
                    pickle.dump(self.scalers, f)
                logger.info(f"Scalers saved to {scalers_path}")
                
            # Save ML model if available
            if self.use_ml and self.ml_model is not None:
                ml_path = os.path.join(directory, f"{ticker}_ml_model.pkl")
                with open(ml_path, 'wb') as f:
                    pickle.dump(self.ml_model, f)
                logger.info(f"ML model saved to {ml_path}")
                
                # Save feature columns
                feat_path = os.path.join(directory, f"{ticker}_features.pkl")
                with open(feat_path, 'wb') as f:
                    pickle.dump(self.feature_columns, f)
                logger.info(f"Feature columns saved to {feat_path}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False

    def load_model(self, directory: str, ticker: str) -> bool:
        """
        Load trained models.

        Args:
            directory: Directory with saved models
            ticker: Stock symbol for filename

        Returns:
            Whether loading was successful
        """
        try:
            import os
            import pickle
            
            # Load LSTM model if available
            lstm_path = os.path.join(directory, f"{ticker}_lstm_model.h5")
            if os.path.exists(lstm_path) and self.use_deep_learning:
                self.dl_model = tf.keras.models.load_model(lstm_path)
                logger.info(f"LSTM model loaded from {lstm_path}")
                
                # Load scalers
                scalers_path = os.path.join(directory, f"{ticker}_scalers.pkl")
                if os.path.exists(scalers_path):
                    with open(scalers_path, 'rb') as f:
                        self.scalers = pickle.load(f)
                    logger.info(f"Scalers loaded from {scalers_path}")
                
            # Load ML model if available
            ml_path = os.path.join(directory, f"{ticker}_ml_model.pkl")
            if os.path.exists(ml_path) and self.use_ml:
                with open(ml_path, 'rb') as f:
                    self.ml_model = pickle.load(f)
                logger.info(f"ML model loaded from {ml_path}")
                
                # Load feature columns
                feat_path = os.path.join(directory, f"{ticker}_features.pkl")
                if os.path.exists(feat_path):
                    with open(feat_path, 'rb') as f:
                        self.feature_columns = pickle.load(f)
                    logger.info(f"Feature columns loaded from {feat_path}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False 