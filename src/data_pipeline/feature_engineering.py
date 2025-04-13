"""
Feature Engineering Module.

This module processes raw market data and news sentiment to create features
for forecasting models and reinforcement learning agents.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import ta  # Technical Analysis library

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Class for creating and transforming features from raw data."""

    def __init__(self):
        """Initialize the feature engineer."""
        logger.info("FeatureEngineer initialized")

    def process_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw market data and add technical indicators.

        Args:
            df: DataFrame with market data (with open, high, low, close, volume columns)

        Returns:
            DataFrame with additional technical indicators
        """
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Make sure the DataFrame is not empty
        if processed_df.empty:
            logger.warning("Empty DataFrame provided to process_market_data")
            return processed_df
            
        # Make sure required columns exist (case insensitive check)
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        df_columns_lower = [col.lower() for col in processed_df.columns]
        
        for col in required_columns:
            if col not in df_columns_lower:
                logger.warning(f"Required column '{col}' not found in DataFrame")
                return processed_df
                
        # Standardize column names (ensure lowercase)
        processed_df.columns = [col.lower() for col in processed_df.columns]
        
        try:
            # Add basic price features
            processed_df['return_1d'] = processed_df['close'].pct_change(1)
            processed_df['return_5d'] = processed_df['close'].pct_change(5)
            processed_df['return_10d'] = processed_df['close'].pct_change(10)
            processed_df['return_20d'] = processed_df['close'].pct_change(20)
            
            # Add rolling statistics
            processed_df['rolling_mean_5d'] = processed_df['close'].rolling(window=5).mean()
            processed_df['rolling_mean_10d'] = processed_df['close'].rolling(window=10).mean()
            processed_df['rolling_mean_20d'] = processed_df['close'].rolling(window=20).mean()
            processed_df['rolling_mean_50d'] = processed_df['close'].rolling(window=50).mean()
            processed_df['rolling_mean_200d'] = processed_df['close'].rolling(window=200).mean()
            
            processed_df['rolling_std_10d'] = processed_df['close'].rolling(window=10).std()
            processed_df['rolling_std_20d'] = processed_df['close'].rolling(window=20).std()
            
            # Add volatility measures
            processed_df['daily_volatility'] = processed_df['return_1d'].rolling(window=20).std()
            processed_df['weekly_volatility'] = processed_df['return_5d'].rolling(window=12).std()
            
            # Add volume features
            processed_df['volume_change_1d'] = processed_df['volume'].pct_change(1)
            processed_df['volume_change_5d'] = processed_df['volume'].pct_change(5)
            processed_df['volume_rolling_mean_5d'] = processed_df['volume'].rolling(window=5).mean()
            processed_df['volume_rolling_mean_20d'] = processed_df['volume'].rolling(window=20).mean()
            
            # Add OHLC-based features
            processed_df['daily_range'] = processed_df['high'] - processed_df['low']
            processed_df['daily_range_pct'] = (processed_df['high'] - processed_df['low']) / processed_df['low']
            processed_df['gap_open'] = (processed_df['open'] - processed_df['close'].shift(1)) / processed_df['close'].shift(1)
            
            # Add technical indicators using TA library
            # Trend indicators
            processed_df['sma_20'] = ta.trend.sma_indicator(processed_df['close'], window=20)
            processed_df['sma_50'] = ta.trend.sma_indicator(processed_df['close'], window=50)
            processed_df['ema_20'] = ta.trend.ema_indicator(processed_df['close'], window=20)
            
            # Add MACD
            macd = ta.trend.MACD(processed_df['close'])
            processed_df['macd'] = macd.macd()
            processed_df['macd_signal'] = macd.macd_signal()
            processed_df['macd_diff'] = macd.macd_diff()
            
            # Add RSI
            processed_df['rsi_14'] = ta.momentum.RSIIndicator(processed_df['close'], window=14).rsi()
            
            # Add Bollinger Bands
            bollinger = ta.volatility.BollingerBands(processed_df['close'], window=20, window_dev=2)
            processed_df['bollinger_high'] = bollinger.bollinger_hband()
            processed_df['bollinger_low'] = bollinger.bollinger_lband()
            processed_df['bollinger_width'] = bollinger.bollinger_wband()
            
            # Add Average True Range (ATR)
            processed_df['atr'] = ta.volatility.AverageTrueRange(
                processed_df['high'], 
                processed_df['low'], 
                processed_df['close'], 
                window=14
            ).average_true_range()
            
            # Add On-Balance Volume (OBV)
            processed_df['obv'] = ta.volume.OnBalanceVolumeIndicator(
                processed_df['close'], 
                processed_df['volume']
            ).on_balance_volume()
            
            # Add Money Flow Index
            processed_df['mfi'] = ta.volume.MFIIndicator(
                processed_df['high'], 
                processed_df['low'], 
                processed_df['close'], 
                processed_df['volume'], 
                window=14
            ).money_flow_index()
            
            # Drop NaN values that might have been introduced
            processed_df = processed_df.fillna(method='bfill')
            
            logger.info("Market data processed successfully")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
            return df  # Return the original DataFrame in case of error

    def add_sentiment_features(
        self, 
        market_df: pd.DataFrame, 
        sentiment_df: pd.DataFrame, 
        date_column: str = 'published_at'
    ) -> pd.DataFrame:
        """
        Add sentiment features to market data.

        Args:
            market_df: DataFrame with processed market data
            sentiment_df: DataFrame with news sentiment data
            date_column: Column name in sentiment_df containing the date

        Returns:
            DataFrame with added sentiment features
        """
        if market_df.empty or sentiment_df.empty:
            logger.warning("Empty DataFrame provided to add_sentiment_features")
            return market_df
            
        try:
            # Make a copy to avoid modifying the original
            result_df = market_df.copy()
            
            # Convert dates in sentiment_df
            sentiment_df[date_column] = pd.to_datetime(sentiment_df[date_column])
            
            # Group sentiment by date and calculate daily aggregates
            daily_sentiment = sentiment_df.groupby(sentiment_df[date_column].dt.date).agg({
                'sentiment_score': ['mean', 'std', 'count', 'min', 'max']
            })
            
            daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns.values]
            daily_sentiment.reset_index(inplace=True)
            daily_sentiment.rename(columns={date_column: 'date'}, inplace=True)
            daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
            
            # Merge with market data
            result_df.reset_index(inplace=True)
            result_df['date'] = pd.to_datetime(result_df['date'])
            result_df = pd.merge(result_df, daily_sentiment, on='date', how='left')
            
            # Create rolling sentiment features
            result_df['sentiment_score_mean_rolling_3d'] = result_df['sentiment_score_mean'].rolling(window=3).mean()
            result_df['sentiment_score_mean_rolling_7d'] = result_df['sentiment_score_mean'].rolling(window=7).mean()
            
            # Fill NaN values
            result_df = result_df.fillna(method='bfill')
            result_df = result_df.fillna(0)  # Fill any remaining NaNs with 0
            
            # Set the index back
            result_df.set_index('date', inplace=True)
            
            logger.info("Sentiment features added successfully")
            return result_df
            
        except Exception as e:
            logger.error(f"Error adding sentiment features: {e}")
            return market_df  # Return the original DataFrame in case of error

    def create_features_for_prediction(
        self, 
        market_df: pd.DataFrame, 
        window_size: int = 10
    ) -> tuple:
        """
        Create features for time-series prediction.

        Args:
            market_df: DataFrame with processed market data
            window_size: Number of time steps to include in each feature vector

        Returns:
            tuple of (X, y) for training a prediction model
        """
        if market_df.empty:
            logger.warning("Empty DataFrame provided to create_features_for_prediction")
            return None, None
            
        try:
            # Make a copy to avoid modifying the original
            df = market_df.copy()
            
            # Select relevant features
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
            
            # Target variable: next day's return
            df['target'] = df['close'].pct_change(1).shift(-1)
            
            # Drop rows with NaN
            df = df.dropna()
            
            # Create sequences
            X = []
            y = []
            
            for i in range(len(df) - window_size):
                X.append(df[feature_columns].iloc[i:i+window_size].values)
                y.append(df['target'].iloc[i+window_size])
                
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Created prediction features with shape X: {X.shape}, y: {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating prediction features: {e}")
            return None, None
            
    def create_stationary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features to be more stationary.

        Args:
            df: DataFrame with market data

        Returns:
            DataFrame with stationary features
        """
        try:
            # Make a copy to avoid modifying the original
            stationary_df = df.copy()
            
            # Log transformation for positive values (prices, volume)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in stationary_df.columns:
                    stationary_df[f'log_{col}'] = np.log(stationary_df[col])
                    
            # Differencing for time series
            for col in ['log_close', 'log_volume']:
                if col in stationary_df.columns:
                    stationary_df[f'diff_{col}'] = stationary_df[col].diff()
                    
            # Relative price features
            if all(col in stationary_df.columns for col in ['close', 'sma_20']):
                stationary_df['price_to_sma20'] = stationary_df['close'] / stationary_df['sma_20'] - 1
                
            if all(col in stationary_df.columns for col in ['close', 'sma_50']):
                stationary_df['price_to_sma50'] = stationary_df['close'] / stationary_df['sma_50'] - 1
                
            if all(col in stationary_df.columns for col in ['sma_20', 'sma_50']):
                stationary_df['sma20_to_sma50'] = stationary_df['sma_20'] / stationary_df['sma_50'] - 1
                
            # Drop rows with NaN values
            stationary_df = stationary_df.dropna()
            
            logger.info("Created stationary features successfully")
            return stationary_df
            
        except Exception as e:
            logger.error(f"Error creating stationary features: {e}")
            return df  # Return the original DataFrame in case of error 