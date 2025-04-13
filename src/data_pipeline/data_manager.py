"""
Data Manager Module.

This module coordinates data flow between data sources, preprocessing, 
and feature engineering for the AlphaQuant system.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
import pickle

from .market_data import MarketDataClient
from .news_sentiment import NewsSentimentAnalyzer
from .feature_engineering import FeatureEngineer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataManager:
    """
    Class for managing data flow in the AlphaQuant system.
    
    This class coordinates the retrieval, processing, and storage of market data,
    news sentiment, and derived features for use in forecasting and decision-making.
    """

    def __init__(self, cache_dir: str = './data/cache'):
        """
        Initialize the data manager.

        Args:
            cache_dir: Directory for caching data
        """
        self.market_data_client = MarketDataClient()
        self.news_sentiment_analyzer = NewsSentimentAnalyzer()
        self.feature_engineer = FeatureEngineer()
        
        # Set up cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataManager initialized (cache_dir: {cache_dir})")

    def get_processed_stock_data(
        self, 
        ticker: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
        interval: str = "1d",
        use_cache: bool = True,
        include_sentiment: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get processed stock data with technical indicators and sentiment.

        Args:
            ticker: Stock symbol
            start_date: Start date (if None, uses period)
            end_date: End date (defaults to today)
            period: Time period (used if start_date is None)
            interval: Data interval (1d, 1h, etc.)
            use_cache: Whether to use cached data if available
            include_sentiment: Whether to include news sentiment features
            force_refresh: Whether to force a refresh of data even if cached

        Returns:
            DataFrame with processed stock data
        """
        cache_file = self.cache_dir / f"{ticker}_processed_{period}_{interval}.pkl"
        
        # Check if cached data exists and is recent
        if use_cache and cache_file.exists() and not force_refresh:
            cache_modified_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            cache_age_hours = (datetime.now() - cache_modified_time).total_seconds() / 3600
            
            # Only use cache if it's less than 24 hours old
            if cache_age_hours < 24:
                logger.info(f"Loading cached data for {ticker} (cache age: {cache_age_hours:.1f} hours)")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        
        # Get raw market data
        try:
            logger.info(f"Fetching market data for {ticker}")
            market_data = self.market_data_client.get_historical_data(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                period=period,
                interval=interval
            )
            
            if market_data.empty:
                logger.warning(f"No market data found for {ticker}")
                return pd.DataFrame()
                
            # Process market data with technical indicators
            processed_data = self.feature_engineer.process_market_data(market_data)
            
            # Add sentiment features if requested
            if include_sentiment:
                company_info = self.market_data_client.get_company_info(ticker)
                company_name = company_info.get('name', ticker)
                
                logger.info(f"Fetching news sentiment for {company_name} ({ticker})")
                news_data = self.news_sentiment_analyzer.get_company_news(
                    company_name=company_name,
                    ticker=ticker,
                    days_back=30,  # Get up to 30 days of news
                    max_articles=50
                )
                
                if not news_data.empty:
                    processed_data = self.feature_engineer.add_sentiment_features(
                        market_df=processed_data,
                        sentiment_df=news_data
                    )
            
            # Cache the processed data
            if use_cache:
                with open(cache_file, 'wb') as f:
                    pickle.dump(processed_data, f)
                logger.info(f"Cached processed data for {ticker}")
                
            return processed_data
            
        except Exception as e:
            logger.error(f"Error getting processed stock data for {ticker}: {e}")
            return pd.DataFrame()

    def get_market_overview(self) -> Dict:
        """
        Get an overview of the current market conditions.

        Returns:
            Dictionary with market overview information
        """
        try:
            # Get market summary
            market_summary = self.market_data_client.get_market_summary()
            
            # Get market sentiment
            market_sentiment = self.news_sentiment_analyzer.get_market_sentiment()
            
            # Combine data
            market_overview = {
                'market_summary': market_summary,
                'market_sentiment': market_sentiment,
                'timestamp': datetime.now().isoformat()
            }
            
            return market_overview
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def get_stock_recommendation(self, ticker: str) -> Dict:
        """
        Get a basic recommendation for a stock.

        Args:
            ticker: Stock symbol

        Returns:
            Dictionary with stock recommendation
        """
        try:
            # Get processed stock data
            stock_data = self.get_processed_stock_data(ticker, period="6mo")
            
            if stock_data.empty:
                return {
                    'ticker': ticker,
                    'recommendation': 'unknown',
                    'confidence': 0.0,
                    'reason': 'No data available',
                    'timestamp': datetime.now().isoformat()
                }
                
            # Get company info
            company_info = self.market_data_client.get_company_info(ticker)
            company_name = company_info.get('name', ticker)
            
            # Get current price
            current_price = stock_data['close'].iloc[-1]
            
            # Simple trend analysis (this would be replaced by the Bayesian network in production)
            short_term_ma = stock_data['rolling_mean_20d'].iloc[-1]
            long_term_ma = stock_data['rolling_mean_50d'].iloc[-1]
            
            # Get sentiment
            sentiment_summary = self.news_sentiment_analyzer.get_sentiment_summary(ticker, company_name)
            
            # Simple rule-based recommendation
            recommendation = 'hold'  # Default to hold
            confidence = 0.5
            reasons = []
            
            # Price above both moving averages
            if current_price > short_term_ma and current_price > long_term_ma:
                if short_term_ma > long_term_ma:  # Short-term MA above long-term MA (bullish)
                    recommendation = 'buy'
                    confidence = 0.7
                    reasons.append("Stock is in an uptrend (price above both 20-day and 50-day moving averages)")
                    
            # Price below both moving averages        
            elif current_price < short_term_ma and current_price < long_term_ma:
                if short_term_ma < long_term_ma:  # Short-term MA below long-term MA (bearish)
                    recommendation = 'sell'
                    confidence = 0.7
                    reasons.append("Stock is in a downtrend (price below both 20-day and 50-day moving averages)")
            
            # Check RSI
            rsi = stock_data['rsi_14'].iloc[-1]
            if rsi > 70:
                if recommendation == 'buy':
                    recommendation = 'hold'
                    confidence = 0.6
                else:
                    recommendation = 'sell'
                    confidence = 0.65
                reasons.append(f"Stock is overbought (RSI: {rsi:.1f} > 70)")
            elif rsi < 30:
                if recommendation == 'sell':
                    recommendation = 'hold'
                    confidence = 0.6
                else:
                    recommendation = 'buy'
                    confidence = 0.65
                reasons.append(f"Stock is oversold (RSI: {rsi:.1f} < 30)")
                
            # Consider sentiment
            sentiment_score = sentiment_summary.get('sentiment_score', 0)
            if sentiment_score > 0.2:
                confidence += 0.1
                reasons.append(f"Positive news sentiment (score: {sentiment_score:.2f})")
            elif sentiment_score < -0.2:
                confidence += 0.1
                reasons.append(f"Negative news sentiment (score: {sentiment_score:.2f})")
                
            # Check if price is near Bollinger Bands
            if 'bollinger_high' in stock_data.columns and 'bollinger_low' in stock_data.columns:
                upper_band = stock_data['bollinger_high'].iloc[-1]
                lower_band = stock_data['bollinger_low'].iloc[-1]
                
                if current_price > upper_band * 0.98:  # Near upper band
                    if recommendation != 'sell':
                        confidence -= 0.1
                    reasons.append("Price is near upper Bollinger Band (potential resistance)")
                elif current_price < lower_band * 1.02:  # Near lower band
                    if recommendation != 'buy':
                        confidence -= 0.1
                    reasons.append("Price is near lower Bollinger Band (potential support)")
            
            # Cap confidence between 0.1 and 0.9
            confidence = max(0.1, min(0.9, confidence))
            
            return {
                'ticker': ticker,
                'company_name': company_name,
                'current_price': current_price,
                'recommendation': recommendation,
                'confidence': confidence,
                'reasons': reasons,
                'technical_indicators': {
                    'rsi': rsi,
                    'short_term_ma': short_term_ma,
                    'long_term_ma': long_term_ma
                },
                'sentiment': sentiment_summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting stock recommendation for {ticker}: {e}")
            return {
                'ticker': ticker,
                'recommendation': 'unknown',
                'confidence': 0.0,
                'reason': f"Error: {str(e)}",
                'timestamp': datetime.now().isoformat()
            }

    def prepare_data_for_model(
        self,
        ticker: str,
        window_size: int = 10,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing prediction models.

        Args:
            ticker: Stock symbol
            window_size: Number of time steps to include in each feature vector
            test_size: Proportion of data to use for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            # Get processed stock data
            stock_data = self.get_processed_stock_data(ticker, period="2y")
            
            if stock_data.empty:
                logger.warning(f"No data available for {ticker}")
                return None, None, None, None
                
            # Create features for prediction
            X, y = self.feature_engineer.create_features_for_prediction(stock_data, window_size)
            
            if X is None or y is None:
                logger.warning(f"Failed to create prediction features for {ticker}")
                return None, None, None, None
                
            # Split into training and test sets
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(f"Prepared data for model: X_train {X_train.shape}, X_test {X_test.shape}")
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data for model: {e}")
            return None, None, None, None
            
    def get_portfolio_data(self, tickers: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks for portfolio analysis.

        Args:
            tickers: List of stock symbols
            period: Time period

        Returns:
            Dictionary mapping tickers to their processed data DataFrames
        """
        portfolio_data = {}
        
        for ticker in tickers:
            try:
                df = self.get_processed_stock_data(ticker, period=period)
                if not df.empty:
                    portfolio_data[ticker] = df
            except Exception as e:
                logger.warning(f"Error getting data for {ticker}: {e}")
                
        return portfolio_data
        
    def save_to_json(self, data: Dict, filename: str) -> bool:
        """
        Save dictionary data to a JSON file.

        Args:
            data: Dictionary to save
            filename: Name of the file

        Returns:
            Whether the operation was successful
        """
        try:
            file_path = self.cache_dir / filename
            
            # Convert any non-serializable objects
            def json_serializer(obj):
                if isinstance(obj, (datetime, np.datetime64)):
                    return obj.isoformat()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f"Type {type(obj)} not serializable")
            
            with open(file_path, 'w') as f:
                json.dump(data, f, default=json_serializer, indent=2)
                
            logger.info(f"Data saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving data to {filename}: {e}")
            return False
            
    def load_from_json(self, filename: str) -> Optional[Dict]:
        """
        Load data from a JSON file.

        Args:
            filename: Name of the file

        Returns:
            Loaded dictionary or None if unsuccessful
        """
        try:
            file_path = self.cache_dir / filename
            
            if not file_path.exists():
                logger.warning(f"File {file_path} does not exist")
                return None
                
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            logger.info(f"Data loaded from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data from {filename}: {e}")
            return None 