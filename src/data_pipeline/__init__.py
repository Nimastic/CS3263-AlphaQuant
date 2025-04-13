"""
AlphaQuant Data Pipeline Module.

This module handles data acquisition, preprocessing, and feature engineering
for market data, news sentiment, and other financial information.
"""

from .market_data import MarketDataClient
from .news_sentiment import NewsSentimentAnalyzer
from .feature_engineering import FeatureEngineer
from .data_manager import DataManager

__all__ = [
    'MarketDataClient',
    'NewsSentimentAnalyzer',
    'FeatureEngineer',
    'DataManager',
] 