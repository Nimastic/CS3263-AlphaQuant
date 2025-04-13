"""
API Routes Module.

This module defines the API routes for the AlphaQuant system.
"""

import os
import logging
from fastapi import APIRouter, HTTPException, Query, Path, Depends
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/api/v1",
    tags=["alphaquant"],
    responses={404: {"description": "Not found"}},
)

# Import data and model modules
from src.data_pipeline.data_manager import DataManager
from src.models.bayesian_forecaster import BayesianForecaster
from src.models.market_predictor import MarketPredictor

# Initialize data manager
data_manager = DataManager(cache_dir="./data/cache")


# Define Pydantic models for request and response validation
class StockRequest(BaseModel):
    """Request model for stock data."""
    ticker: str = Field(..., description="Stock symbol")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    period: Optional[str] = Field("1y", description="Time period (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y')")
    interval: Optional[str] = Field("1d", description="Data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')")


class StockResponse(BaseModel):
    """Response model for stock data."""
    ticker: str
    start_date: str
    end_date: str
    period: str
    interval: str
    data_points: int
    columns: List[str]
    data: Dict[str, List[Any]]
    timestamp: str


class PredictionRequest(BaseModel):
    """Request model for stock prediction."""
    ticker: str = Field(..., description="Stock symbol")
    days: int = Field(5, description="Number of days to predict")
    use_sentiment: bool = Field(True, description="Whether to include sentiment analysis")


class PredictionResponse(BaseModel):
    """Response model for stock prediction."""
    ticker: str
    prediction_type: str
    forecast: Dict[str, Any]
    current_price: float
    timestamp: str


class RecommendationRequest(BaseModel):
    """Request model for stock recommendation."""
    ticker: str = Field(..., description="Stock symbol")
    risk_profile: Optional[str] = Field("moderate", description="User risk profile (conservative, moderate, aggressive)")


class RecommendationResponse(BaseModel):
    """Response model for stock recommendation."""
    ticker: str
    company_name: str
    current_price: float
    recommendation: str
    confidence: float
    reasons: List[str]
    technical_indicators: Dict[str, Any]
    sentiment: Dict[str, Any]
    timestamp: str


class PortfolioRequest(BaseModel):
    """Request model for portfolio optimization."""
    tickers: List[str] = Field(..., description="List of stock symbols")
    initial_investment: float = Field(10000.0, description="Initial investment amount")
    risk_profile: Optional[str] = Field("moderate", description="User risk profile (conservative, moderate, aggressive)")


class PortfolioResponse(BaseModel):
    """Response model for portfolio optimization."""
    tickers: List[str]
    weights: List[float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    timestamp: str


class MarketOverviewResponse(BaseModel):
    """Response model for market overview."""
    market_summary: Dict[str, Any]
    market_sentiment: Dict[str, Any]
    timestamp: str


@router.get("/stock/{ticker}", response_model=StockResponse)
async def get_stock_data(
    ticker: str = Path(..., description="Stock symbol"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    period: str = Query("1y", description="Time period"),
    interval: str = Query("1d", description="Data interval"),
    include_indicators: bool = Query(True, description="Whether to include technical indicators")
):
    """Get historical stock data with optional technical indicators."""
    try:
        logger.info(f"Getting stock data for {ticker}")
        
        # Get stock data
        stock_data = data_manager.get_processed_stock_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            period=period,
            interval=interval,
            include_sentiment=False  # Don't include sentiment for basic data request
        )
        
        if stock_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")
            
        # Convert to dict for response
        data_dict = {}
        for column in stock_data.columns:
            # Skip technical indicators if not requested
            if not include_indicators and column not in ['open', 'high', 'low', 'close', 'volume']:
                continue
            data_dict[column] = stock_data[column].tolist()
            
        # Get date range
        start_date_str = stock_data.index[0].strftime('%Y-%m-%d')
        end_date_str = stock_data.index[-1].strftime('%Y-%m-%d')
        
        # Create response
        response = {
            "ticker": ticker,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "period": period,
            "interval": interval,
            "data_points": len(stock_data),
            "columns": list(data_dict.keys()),
            "data": data_dict,
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting stock data for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    """Generate stock price prediction using Bayesian forecasting."""
    try:
        ticker = request.ticker
        days = request.days
        use_sentiment = request.use_sentiment
        
        logger.info(f"Generating prediction for {ticker} ({days} days)")
        
        # Get stock data
        stock_data = data_manager.get_processed_stock_data(
            ticker=ticker,
            period="1y",
            include_sentiment=use_sentiment
        )
        
        if stock_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for ticker {ticker}")
            
        # Create and fit Bayesian forecaster
        forecaster = BayesianForecaster()
        forecast_result = forecaster.fit_model_for_ticker(stock_data, ticker, forecast_days=days)
        
        if 'error' in forecast_result:
            raise HTTPException(status_code=500, detail=forecast_result['error'])
            
        # Get current price
        current_price = stock_data['close'].iloc[-1]
        
        # Create response
        response = {
            "ticker": ticker,
            "prediction_type": "bayesian",
            "forecast": forecast_result['forecast'],
            "current_price": current_price,
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating prediction for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendation(request: RecommendationRequest):
    """Get investment recommendation for a stock."""
    try:
        ticker = request.ticker
        risk_profile = request.risk_profile
        
        logger.info(f"Getting recommendation for {ticker} (risk profile: {risk_profile})")
        
        # Get recommendation from data manager
        recommendation = data_manager.get_stock_recommendation(ticker)
        
        if 'error' in recommendation:
            raise HTTPException(status_code=500, detail=recommendation['error'])
            
        # Create response
        response = recommendation
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting recommendation for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/portfolio/optimize", response_model=PortfolioResponse)
async def optimize_portfolio(request: PortfolioRequest):
    """Optimize portfolio allocation based on user preferences."""
    try:
        tickers = request.tickers
        initial_investment = request.initial_investment
        risk_profile = request.risk_profile
        
        logger.info(f"Optimizing portfolio for {len(tickers)} stocks (risk profile: {risk_profile})")
        
        # Get data for all tickers
        portfolio_data = data_manager.get_portfolio_data(tickers, period="1y")
        
        if not portfolio_data:
            raise HTTPException(status_code=404, detail="No data found for specified tickers")
            
        # Convert risk profile to risk aversion parameter
        risk_aversion = 1.0  # Default (moderate)
        if risk_profile == "conservative":
            risk_aversion = 2.0
        elif risk_profile == "aggressive":
            risk_aversion = 0.5
            
        # Simple portfolio optimization (mean-variance optimization)
        # For now, this is a simplified implementation
        # In a real system, this would use more sophisticated optimization techniques
        
        # Calculate expected returns and covariance matrix
        returns = {}
        for ticker, data in portfolio_data.items():
            returns[ticker] = data['return_1d'].mean()
            
        # Simplified weights calculation (equal weight for now)
        weights = [1.0 / len(tickers) for _ in tickers]
        
        # Calculate portfolio statistics
        expected_return = sum(returns[ticker] * weight for ticker, weight in zip(tickers, weights))
        expected_risk = 0.02  # Simplified risk calculation
        sharpe_ratio = expected_return / expected_risk
        
        # Create response
        response = {
            "tickers": tickers,
            "weights": weights,
            "expected_return": expected_return * 252,  # Annualized
            "expected_risk": expected_risk * (252 ** 0.5),  # Annualized
            "sharpe_ratio": sharpe_ratio,
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error optimizing portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market/overview", response_model=MarketOverviewResponse)
async def get_market_overview():
    """Get an overview of the current market conditions."""
    try:
        logger.info("Getting market overview")
        
        # Get market overview from data manager
        overview = data_manager.get_market_overview()
        
        if 'error' in overview:
            raise HTTPException(status_code=500, detail=overview['error'])
            
        # Create response
        response = overview
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting market overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tickers/search")
async def search_tickers(query: str = Query(..., description="Search query")):
    """Search for stock tickers based on query."""
    try:
        logger.info(f"Searching for tickers: {query}")
        
        # For demonstration purposes, return some hardcoded results
        # In a real system, this would query a database or external API
        tickers = [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
            {"symbol": "GOOGL", "name": "Alphabet Inc."},
            {"symbol": "AMZN", "name": "Amazon.com, Inc."},
            {"symbol": "META", "name": "Meta Platforms, Inc."},
            {"symbol": "TSLA", "name": "Tesla, Inc."},
            {"symbol": "NVDA", "name": "NVIDIA Corporation"},
            {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
            {"symbol": "V", "name": "Visa Inc."},
            {"symbol": "JNJ", "name": "Johnson & Johnson"}
        ]
        
        # Filter tickers based on query
        filtered_tickers = [
            ticker for ticker in tickers
            if query.lower() in ticker["symbol"].lower() or query.lower() in ticker["name"].lower()
        ]
        
        return {"results": filtered_tickers}
        
    except Exception as e:
        logger.error(f"Error searching for tickers: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 