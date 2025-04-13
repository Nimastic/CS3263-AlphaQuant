"""
Market Data Client Module.

This module provides interfaces to fetch financial market data from various APIs.
"""

import os
import pandas as pd
import yfinance as yf
from polygon import RESTClient
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataClient:
    """Client for retrieving financial market data from various sources."""

    def __init__(self, use_polygon: bool = True, use_yahoo: bool = True):
        """
        Initialize the market data client.

        Args:
            use_polygon: Whether to use Polygon.io API
            use_yahoo: Whether to use Yahoo Finance API
        """
        self.use_polygon = use_polygon
        self.use_yahoo = use_yahoo
        
        if use_polygon:
            polygon_api_key = os.getenv("POLYGON_API_KEY")
            if not polygon_api_key:
                logger.warning("POLYGON_API_KEY not found in environment variables")
                self.use_polygon = False
            else:
                self.polygon_client = RESTClient(polygon_api_key)
        
        logger.info(f"MarketDataClient initialized (Polygon: {use_polygon}, Yahoo: {use_yahoo})")

    def get_historical_data(
        self, 
        ticker: str, 
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical market data for a specific ticker.

        Args:
            ticker: Stock symbol
            start_date: Start date (if None, uses period)
            end_date: End date (defaults to today)
            period: Time period (used if start_date is None)
            interval: Data interval (1d, 1h, etc.)

        Returns:
            DataFrame with historical market data
        """
        if not end_date:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        if not start_date and not period:
            period = "1y"
            
        if start_date:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            # Convert period to start_date
            if period.endswith("d"):
                days = int(period[:-1])
                start_date = end_date - timedelta(days=days)
            elif period.endswith("w"):
                weeks = int(period[:-1])
                start_date = end_date - timedelta(weeks=weeks)
            elif period.endswith("mo"):
                months = int(period[:-2])
                start_date = end_date - timedelta(days=months*30)
            elif period.endswith("y"):
                years = int(period[:-1])
                start_date = end_date - timedelta(days=years*365)
            else:
                start_date = end_date - timedelta(days=365)  # Default to 1 year
        
        # Format dates for API calls
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # First try Polygon if enabled
        if self.use_polygon:
            try:
                return self._get_data_from_polygon(ticker, start_str, end_str, interval)
            except Exception as e:
                logger.warning(f"Failed to get data from Polygon: {e}")
                # Fall back to Yahoo Finance if both are enabled
                if self.use_yahoo:
                    logger.info("Falling back to Yahoo Finance")
                else:
                    raise
        
        # Use Yahoo Finance
        if self.use_yahoo:
            return self._get_data_from_yahoo(ticker, start_str, end_str, interval)
        
        raise ValueError("No data source available (both Polygon and Yahoo Finance are disabled)")

    def _get_data_from_polygon(self, ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Fetch data from Polygon.io API."""
        # Convert interval to Polygon format
        if interval == "1d":
            timespan = "day"
            multiplier = 1
        elif interval == "1h":
            timespan = "hour"
            multiplier = 1
        elif interval == "1m":
            timespan = "minute"
            multiplier = 1
        else:
            timespan = "day"
            multiplier = 1
        
        # Fetch data
        aggs = self.polygon_client.get_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start_date,
            to=end_date,
            limit=50000
        )
        
        # Convert to DataFrame
        data = []
        for agg in aggs:
            data.append({
                'timestamp': agg.timestamp,
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume,
                'vwap': getattr(agg, 'vwap', None),
                'transactions': getattr(agg, 'transactions', None)
            })
        
        if not data:
            raise ValueError(f"No data returned from Polygon for {ticker}")
            
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.index.name = 'Date'
        
        return df

    def _get_data_from_yahoo(self, ticker: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance API."""
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data returned from Yahoo Finance for {ticker}")
            
        # Standardize column names to match Polygon format
        df.index.name = 'Date'
        df.columns = [col.lower() for col in df.columns]
        
        return df

    def get_current_price(self, ticker: str) -> float:
        """
        Get the current price for a ticker.

        Args:
            ticker: Stock symbol

        Returns:
            Current price
        """
        if self.use_polygon:
            try:
                last_trade = self.polygon_client.get_last_trade(ticker)
                return last_trade.price
            except Exception as e:
                logger.warning(f"Failed to get current price from Polygon: {e}")
                if self.use_yahoo:
                    logger.info("Falling back to Yahoo Finance")
                else:
                    raise
        
        if self.use_yahoo:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            if not data.empty:
                return data['Close'].iloc[-1]
            raise ValueError(f"No data returned from Yahoo Finance for {ticker}")
        
        raise ValueError("No data source available")

    def get_market_summary(self) -> Dict:
        """
        Get a summary of the overall market.

        Returns:
            Dictionary with market summary information
        """
        # Market indices
        indices = ['SPY', 'QQQ', 'DIA', 'IWM']  # S&P 500, Nasdaq, Dow Jones, Russell 2000
        
        summary = {}
        for idx in indices:
            if self.use_yahoo:
                try:
                    data = yf.Ticker(idx).history(period="5d")
                    if not data.empty:
                        prev_close = data['Close'].iloc[-2]
                        current = data['Close'].iloc[-1]
                        change_pct = (current - prev_close) / prev_close * 100
                        
                        summary[idx] = {
                            'current': current,
                            'change_pct': change_pct,
                            'volume': data['Volume'].iloc[-1]
                        }
                except Exception as e:
                    logger.warning(f"Failed to get data for {idx}: {e}")
        
        return summary

    def get_company_info(self, ticker: str) -> Dict:
        """
        Get company information for a ticker.

        Args:
            ticker: Stock symbol

        Returns:
            Dictionary with company information
        """
        if self.use_yahoo:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Extract relevant information
                company_info = {
                    'name': info.get('shortName', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', ''),
                    'market_cap': info.get('marketCap', None),
                    'pe_ratio': info.get('trailingPE', None),
                    'dividend_yield': info.get('dividendYield', None) * 100 if info.get('dividendYield') else None,
                    'beta': info.get('beta', None),
                    'description': info.get('longBusinessSummary', '')
                }
                
                return company_info
            except Exception as e:
                logger.warning(f"Failed to get company info for {ticker}: {e}")
                
        return {}

    def get_multiple_tickers(
        self, 
        tickers: List[str], 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple tickers.

        Args:
            tickers: List of stock symbols
            start_date: Start date
            end_date: End date
            period: Time period
            interval: Data interval

        Returns:
            Dictionary mapping tickers to their historical data DataFrames
        """
        results = {}
        for ticker in tickers:
            try:
                df = self.get_historical_data(ticker, start_date, end_date, period, interval)
                results[ticker] = df
            except Exception as e:
                logger.warning(f"Failed to get data for {ticker}: {e}")
                
        return results 