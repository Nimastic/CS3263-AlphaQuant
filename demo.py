"""
AlphaQuant Demo Script (Real Data Version).

This script demonstrates the functionality of the AlphaQuant system,
using real market data and news.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import json
import requests
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("alphaquant_demo")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()


class DataManager:
    """Data manager for fetching and processing market data."""
    
    def __init__(self, cache_dir="data/cache", output_dir="output"):
        """Initialize the data manager."""
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        
        # Create directories if they don't exist
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized DataManager with cache_dir={cache_dir}")
    
    def get_stock_data(self, ticker, period="1y", interval="1d"):
        """Get real stock data using yfinance."""
        try:
            # Use yfinance for real data
            import yfinance as yf
            logger.info(f"Fetching data for {ticker} from Yahoo Finance")
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            
            # Basic technical indicators
            if not data.empty:
                # Calculate simple moving averages
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                
                # Calculate daily returns
                data['Returns'] = data['Close'].pct_change()
                
                # Calculate volatility (20-day rolling standard deviation of returns)
                data['Volatility'] = data['Returns'].rolling(window=20).std()
                
                logger.info(f"Successfully processed data for {ticker}")
                return data
            else:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()
                
        except ImportError as e:
            logger.error(f"Error importing yfinance: {e}")
            logger.error("Please install yfinance: pip install yfinance")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_market_sentiment(self):
        """Get market sentiment data from real news sources."""
        try:
            # Use NewsAPI for real news
            news_api_key = os.getenv("NEWS_API_KEY", "")
            
            if not news_api_key:
                logger.error("No NEWS_API_KEY found in environment variables")
                return {"error": "No API key available"}
            
            # Fetch financial news from News API
            url = f"https://newsapi.org/v2/everything?q=stock+market+finance&sortBy=publishedAt&apiKey={news_api_key}&pageSize=15&language=en"
            response = requests.get(url)
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch news. Status code: {response.status_code}")
                return {"error": f"API error: {response.status_code}"}
            
            news_data = response.json()
            
            if "articles" not in news_data or not news_data["articles"]:
                logger.error("No articles found in news API response")
                return {"error": "No articles found"}
            
            # Process articles and calculate sentiment
            articles = news_data["articles"]
            headlines = []
            
            positive_words = ["rise", "gain", "growth", "higher", "surge", "rally", "positive", "bullish", "up", "increase", "beats", "record", "strong"]
            negative_words = ["fall", "drop", "decline", "lower", "plunge", "crash", "negative", "bearish", "down", "decrease", "miss", "weak", "trouble", "worries", "fear"]
            
            total_sentiment = 0
            
            for article in articles[:10]:  # Process top 10 articles
                title = article["title"]
                
                # Simple sentiment calculation
                sentiment = 0
                for word in positive_words:
                    if word.lower() in title.lower():
                        sentiment += 0.2
                for word in negative_words:
                    if word.lower() in title.lower():
                        sentiment -= 0.2
                
                # Cap sentiment between -1 and 1
                sentiment = max(min(sentiment, 1.0), -1.0)
                
                # Format publication date
                pub_date = ""
                if "publishedAt" in article:
                    try:
                        date_obj = datetime.strptime(article["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
                        pub_date = date_obj.strftime("%Y-%m-%d %H:%M")
                    except:
                        pub_date = article.get("publishedAt", "")
                
                headlines.append({
                    "title": title,
                    "sentiment": sentiment,
                    "url": article["url"],
                    "source": article.get("source", {}).get("name", "Unknown Source"),
                    "date": pub_date
                })
                
                total_sentiment += sentiment
            
            # Calculate average sentiment
            avg_sentiment = total_sentiment / len(headlines) if headlines else 0
            
            # Determine sentiment label
            if avg_sentiment > 0.3:
                sentiment_label = "positive"
            elif avg_sentiment < -0.3:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            return {
                "market_sentiment": {
                    "market_sentiment_score": avg_sentiment,
                    "market_sentiment_label": sentiment_label,
                    "articles_count": len(news_data["articles"]),
                    "top_headlines": headlines
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching market sentiment: {e}")
            return {"error": str(e)}
    
    def generate_stock_recommendation(self, ticker, data):
        """Generate a stock recommendation based on technical indicators."""
        if data.empty:
            return {
                "ticker": ticker,
                "recommendation": "neutral",
                "confidence": 0.5,
                "reasons": ["Insufficient data available"]
            }
        
        # Recommendation logic based on moving averages and other indicators
        last_close = data['Close'].iloc[-1]
        last_sma20 = data['SMA_20'].iloc[-1]
        last_sma50 = data['SMA_50'].iloc[-1]
        
        # Calculate RSI (Relative Strength Index) for additional signal
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        recommendation = "neutral"
        confidence = 0.5
        reasons = []
        
        # Moving Average signals
        if last_close > last_sma20 and last_sma20 > last_sma50:
            recommendation = "buy"
            confidence = 0.7
            reasons.append("Price is above both 20-day and 50-day moving averages")
            reasons.append("20-day moving average is above 50-day moving average (bullish crossover)")
        elif last_close < last_sma20 and last_sma20 < last_sma50:
            recommendation = "sell"
            confidence = 0.7
            reasons.append("Price is below both 20-day and 50-day moving averages")
            reasons.append("20-day moving average is below 50-day moving average (bearish crossover)")
        elif last_close > last_sma20:
            recommendation = "hold"
            confidence = 0.6
            reasons.append("Price is above 20-day moving average")
            reasons.append("Short-term momentum is positive")
        else:
            recommendation = "hold"
            confidence = 0.6
            reasons.append("Market conditions are mixed")
            reasons.append("Recommend waiting for clearer signals")
        
        # RSI signals - overbought/oversold
        if not np.isnan(current_rsi):
            if current_rsi > 70:
                if recommendation == "buy":
                    recommendation = "hold"
                    confidence = 0.6
                    reasons.append(f"RSI is high ({current_rsi:.1f}), indicating potential overbought conditions")
                elif recommendation == "sell":
                    confidence = 0.8
                    reasons.append(f"RSI is high ({current_rsi:.1f}), confirming overbought conditions")
            elif current_rsi < 30:
                if recommendation == "sell":
                    recommendation = "hold"
                    confidence = 0.6
                    reasons.append(f"RSI is low ({current_rsi:.1f}), indicating potential oversold conditions")
                elif recommendation == "buy":
                    confidence = 0.8
                    reasons.append(f"RSI is low ({current_rsi:.1f}), confirming oversold conditions")
            else:
                reasons.append(f"RSI is neutral at {current_rsi:.1f}")
        
        # Add volatility assessment
        recent_volatility = data['Volatility'].iloc[-1]
        avg_volatility = data['Volatility'].mean()
        
        if recent_volatility > 1.5 * avg_volatility:
            reasons.append(f"Caution: {ticker} is experiencing higher than normal volatility")
        
        return {
            "ticker": ticker,
            "recommendation": recommendation,
            "confidence": confidence,
            "reasons": reasons
        }
    
    def generate_market_map(self, tickers, output_path=None):
        """Generate a market map with stock performance and forecasts."""
        try:
            # Get data for all tickers
            all_data = {}
            for ticker in tickers:
                data = self.get_stock_data(ticker, period="6mo")
                if not data.empty:
                    all_data[ticker] = data
            
            if not all_data:
                logger.error("No data available for market map")
                return False
                
            # Create figure for the market map
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Colors for sector/industry categories
            colors = {
                "AAPL": "royalblue",   # Tech
                "MSFT": "royalblue",   # Tech
                "GOOGL": "royalblue",  # Tech
                "AMZN": "forestgreen", # Consumer
                "META": "royalblue",   # Tech
                "TSLA": "firebrick",   # Auto
                "JPM": "purple",       # Finance
                "JNJ": "orange",       # Healthcare
                "PG": "forestgreen",   # Consumer
                "V": "purple",         # Finance
            }
            
            # Prepare data for plotting
            x_values = []  # For volatility
            y_values = []  # For YTD return
            sizes = []     # For trading volume
            labels = []    # For ticker names
            colors_list = []  # For sector colors
            
            for ticker, data in all_data.items():
                # Calculate % change since start
                first_close = data['Close'].iloc[0]
                last_close = data['Close'].iloc[-1]
                pct_change = (last_close - first_close) / first_close * 100
                
                # Get average trading volume as proxy for market cap
                avg_volume = data['Volume'].mean()
                
                # Calculate 30-day volatility for x-axis
                volatility = data['Returns'].rolling(window=30).std().iloc[-1] * 100
                
                x_values.append(volatility)
                y_values.append(pct_change)
                sizes.append(avg_volume / 1e6)  # Scale down for bubble size
                labels.append(ticker)
                colors_list.append(colors.get(ticker, "gray"))
            
            # Create the scatter plot
            scatter = ax.scatter(
                x_values, 
                y_values, 
                s=sizes,
                c=colors_list,
                alpha=0.7
            )
            
            # Add ticker labels to each point
            for i, ticker in enumerate(labels):
                ax.annotate(
                    ticker, 
                    (x_values[i], y_values[i]),
                    xytext=(5, 5),
                    textcoords='offset points'
                )
            
            # Add a horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add plot labels and title
            ax.set_xlabel('30-Day Volatility (%)')
            ax.set_ylabel('Price Change (%)')
            ax.set_title('Market Map: Performance vs. Volatility')
            ax.grid(True, alpha=0.3)
            
            # Add forecast arrows for a few stocks
            forecaster = BayesianForecaster()
            forecast_tickers = ["AAPL", "MSFT", "GOOGL"]
            
            for ticker in forecast_tickers:
                if ticker in all_data:
                    # Generate forecast
                    forecast_result = forecaster.forecast(all_data[ticker], ticker, days=30)
                    
                    if 'forecast' in forecast_result:
                        forecast = forecast_result['forecast']
                        
                        # Calculate predicted % change
                        current_price = all_data[ticker]['Close'].iloc[-1]
                        forecast_price = forecast['forecast_mean'][-1]
                        forecast_pct_change = (forecast_price - current_price) / current_price * 100
                        
                        # Find index of this ticker
                        idx = labels.index(ticker)
                        
                        # Draw arrow from current position to forecast position
                        ax.annotate(
                            '', 
                            xy=(x_values[idx], y_values[idx] + forecast_pct_change),
                            xytext=(x_values[idx], y_values[idx]),
                            arrowprops=dict(
                                facecolor=colors.get(ticker, 'gray'),
                                shrink=0.05,
                                width=2,
                                headwidth=8,
                                alpha=0.7
                            )
                        )
                        
                        # Add forecast label
                        ax.annotate(
                            f"{ticker} Forecast",
                            xy=(x_values[idx], y_values[idx] + forecast_pct_change),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontweight='bold'
                        )
            
            # Add legend for market sectors
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', markersize=10, label='Technology'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='forestgreen', markersize=10, label='Consumer'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Finance'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Healthcare'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='firebrick', markersize=10, label='Auto')
            ]
            ax.legend(handles=legend_elements, loc='upper left')
            
            plt.tight_layout()
            
            # Save the figure if path is provided
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Market map saved to {output_path}")
            
            plt.close()
            return True
            
        except Exception as e:
            logger.error(f"Error generating market map: {e}")
            return False


class BayesianForecaster:
    """Bayesian forecaster for price predictions."""
    
    def __init__(self):
        """Initialize the forecaster."""
        self.forecast_data = None
        logger.info("Initialized BayesianForecaster")
    
    def forecast(self, data, ticker, days=5):
        """Generate a Bayesian forecast."""
        if data.empty:
            return {
                "error": "No data available for forecasting"
            }
            
        # Extract the closing prices
        prices = data['Close'].dropna()
        
        # Calculate mean and standard deviation of returns
        returns = prices.pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate forecast
        last_price = prices.iloc[-1]
        forecast_dates = pd.date_range(start=prices.index[-1] + pd.Timedelta(days=1), periods=days, freq='B')
        
        # Generate multiple scenarios with Bayesian approach
        np.random.seed(int(time.time()))  # Use current time for different results each run
        scenarios = 1000
        all_paths = np.zeros((scenarios, days))
        
        for i in range(scenarios):
            # Sample mean and std from posterior distributions
            sample_mean = np.random.normal(mean_return, mean_return/10)
            sample_std = std_return * np.random.gamma(shape=9, scale=1/10)
            
            path = [last_price]
            for _ in range(days):
                # Sample from normal distribution with sampled parameters
                daily_return = np.random.normal(sample_mean, sample_std)
                next_price = path[-1] * (1 + daily_return)
                path.append(next_price)
            all_paths[i] = path[1:]  # Exclude the starting price
            
        # Calculate statistics
        mean_forecast = np.mean(all_paths, axis=0)
        lower_bound = np.percentile(all_paths, 5, axis=0)
        upper_bound = np.percentile(all_paths, 95, axis=0)
        
        # Store forecast data for plotting
        self.forecast_data = {
            'last_price': last_price,
            'dates': forecast_dates,
            'mean': mean_forecast,
            'lower': lower_bound,
            'upper': upper_bound
        }
        
        # Format the response
        formatted_forecast = {
            'ticker': ticker,
            'last_price': float(last_price),
            'forecast_start_date': forecast_dates[0].strftime('%Y-%m-%d'),
            'forecast_end_date': forecast_dates[-1].strftime('%Y-%m-%d'),
            'forecast_mean': [float(x) for x in mean_forecast],
            'forecast_lower_bound': [float(x) for x in lower_bound],
            'forecast_upper_bound': [float(x) for x in upper_bound]
        }
        
        return {'forecast': formatted_forecast}
    
    def plot_forecast(self, historical_data, save_path=None):
        """Plot the forecast with historical data."""
        if self.forecast_data is None:
            logger.warning("No forecast data available for plotting")
            return False
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot historical data (last 30 days)
        historical_prices = historical_data['Close'].iloc[-30:]
        plt.plot(historical_prices.index, historical_prices, label='Historical', color='blue')
        
        # Plot forecast
        forecast_dates = self.forecast_data['dates']
        plt.plot(forecast_dates, self.forecast_data['mean'], label='Forecast', color='red')
        
        # Plot confidence interval
        plt.fill_between(
            forecast_dates,
            self.forecast_data['lower'],
            self.forecast_data['upper'],
            color='red',
            alpha=0.2,
            label='90% Confidence Interval'
        )
        
        # Add labels and legend
        plt.title('Price Forecast with Uncertainty')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Forecast plot saved to {save_path}")
            
        plt.close()
        return True


def select_portfolio_based_on_recommendations(recommendations, tickers, min_confidence=0.6):
    """Select portfolio tickers and weights based on stock recommendations.
    
    Args:
        recommendations: Dictionary of stock recommendations
        tickers: List of available tickers
        min_confidence: Minimum confidence threshold
        
    Returns:
        Tuple of (selected_tickers, weights)
    """
    selected_tickers = []
    scores = []
    
    for ticker in tickers:
        if ticker not in recommendations:
            continue
            
        rec = recommendations[ticker]
        
        # Skip if confidence is too low
        if rec['confidence'] < min_confidence:
            continue
            
        # Calculate a score based on recommendation and confidence
        score = 0
        if rec['recommendation'] == 'buy':
            score = rec['confidence'] * 2
        elif rec['recommendation'] == 'hold':
            score = rec['confidence']
        elif rec['recommendation'] == 'sell':
            continue  # Skip sell recommendations
            
        if score > 0:
            selected_tickers.append(ticker)
            scores.append(score)
    
    # If no stocks meet criteria, return default
    if not selected_tickers:
        logger.warning("No stocks met recommendation criteria, using default portfolio")
        return ["AAPL", "MSFT", "GOOGL"], [1/3, 1/3, 1/3]
    
    # Calculate weights proportional to scores
    total_score = sum(scores)
    weights = [score/total_score for score in scores]
    
    logger.info(f"Selected portfolio based on recommendations: {list(zip(selected_tickers, weights))}")
    return selected_tickers, weights


def run_portfolio_backtest(stock_data, tickers, lookback_period=120):
    """Run a backtest comparing recommendation-based portfolio against equal-weight.
    
    Args:
        stock_data: Dictionary of stock dataframes
        tickers: List of available tickers
        lookback_period: Number of days to use for backtest (default: 120 trading days)
        
    Returns:
        Dictionary with backtest results
    """
    logger.info(f"Running portfolio backtest with {lookback_period} days lookback period")
    
    # Ensure we have enough data for all tickers
    valid_tickers = []
    for ticker in tickers:
        if ticker in stock_data and not stock_data[ticker].empty:
            if len(stock_data[ticker]) >= lookback_period:
                valid_tickers.append(ticker)
    
    if len(valid_tickers) < 3:
        logger.error("Not enough data for backtest")
        return {"error": "Not enough data for backtest"}
    
    # Set up backtest parameters
    test_periods = lookback_period // 20  # Rebalance every 20 trading days
    benchmark_tickers = valid_tickers[:5]  # Take first 5 valid tickers for benchmark
    
    # Initialize results
    rec_portfolio_values = []  # Recommendation-based portfolio
    equal_portfolio_values = []  # Equal weight portfolio
    dates = []
    
    # Get common dates for all valid tickers
    common_dates = None
    for ticker in valid_tickers:
        if common_dates is None:
            common_dates = set(stock_data[ticker].index[-lookback_period:])
        else:
            common_dates = common_dates.intersection(set(stock_data[ticker].index[-lookback_period:]))
    
    if not common_dates:
        logger.error("No common dates found for backtest")
        return {"error": "No common dates found for backtest"}
    
    common_dates = sorted(list(common_dates))
    
    # Run the backtest
    test_start_indices = list(range(0, lookback_period, 20))
    
    # Initialize portfolio values
    rec_value = 100.0
    equal_value = 100.0
    
    for i, start_idx in enumerate(test_start_indices[:-1]):
        period_dates = common_dates[start_idx:test_start_indices[i+1]]
        period_start_date = period_dates[0]
        
        # Generate recommendations at the start of each period
        all_recommendations = {}
        
        for ticker in valid_tickers:
            # Use data up to the current period start date to generate recommendation
            historical_data = stock_data[ticker]
            historical_data = historical_data[historical_data.index <= period_start_date]
            
            if len(historical_data) >= 60:  # Need enough data for indicators
                # Create a temporary DataManager to get recommendations
                temp_manager = DataManager()
                recommendation = temp_manager.generate_stock_recommendation(ticker, historical_data)
                all_recommendations[ticker] = recommendation
        
        # Select portfolios
        rec_tickers, rec_weights = select_portfolio_based_on_recommendations(
            all_recommendations, valid_tickers, min_confidence=0.6
        )
        
        equal_tickers = benchmark_tickers
        equal_weights = [1.0/len(equal_tickers)] * len(equal_tickers)
        
        logger.info(f"Period {i+1}: Rec portfolio: {rec_tickers}")
        
        # Calculate returns for this period
        rec_period_return = 0
        equal_period_return = 0
        
        # Get starting and ending prices
        for j, ticker in enumerate(rec_tickers):
            ticker_data = stock_data[ticker]
            start_price = ticker_data.loc[ticker_data.index == period_dates[0], 'Close'].iloc[0]
            end_price = ticker_data.loc[ticker_data.index == period_dates[-1], 'Close'].iloc[0]
            ticker_return = (end_price / start_price) - 1
            rec_period_return += ticker_return * rec_weights[j]
        
        for j, ticker in enumerate(equal_tickers):
            ticker_data = stock_data[ticker]
            start_price = ticker_data.loc[ticker_data.index == period_dates[0], 'Close'].iloc[0]
            end_price = ticker_data.loc[ticker_data.index == period_dates[-1], 'Close'].iloc[0]
            ticker_return = (end_price / start_price) - 1
            equal_period_return += ticker_return * equal_weights[j]
        
        # Update portfolio values
        rec_value *= (1 + rec_period_return)
        equal_value *= (1 + equal_period_return)
        
        # Track values at the end of each period
        dates.append(period_dates[-1])
        rec_portfolio_values.append(rec_value)
        equal_portfolio_values.append(equal_value)
        
        logger.info(f"Period {i+1} returns: Recommended: {rec_period_return:.2%}, Equal-weight: {equal_period_return:.2%}")
    
    # Calculate performance metrics
    rec_total_return = (rec_value / 100.0) - 1
    equal_total_return = (equal_value / 100.0) - 1
    
    # Annualize returns (approximate)
    days_in_test = (common_dates[-1] - common_dates[0]).days
    years = max(days_in_test / 365, 0.1)  # Avoid division by very small numbers
    
    rec_annual_return = (1 + rec_total_return) ** (1 / years) - 1
    equal_annual_return = (1 + equal_total_return) ** (1 / years) - 1
    
    results = {
        "dates": [d.strftime('%Y-%m-%d') for d in dates],
        "recommended_portfolio": {
            "values": rec_portfolio_values,
            "total_return": rec_total_return,
            "annual_return": rec_annual_return
        },
        "equal_weight_portfolio": {
            "values": equal_portfolio_values,
            "total_return": equal_total_return,
            "annual_return": equal_annual_return
        },
        "benchmark_tickers": benchmark_tickers,
        "test_periods": test_periods,
        "days_in_test": days_in_test
    }
    
    # Plot backtest results
    plt.figure(figsize=(12, 6))
    plt.plot(dates, rec_portfolio_values, label=f'Recommendation-Based Portfolio (Return: {rec_total_return:.2%})', linewidth=2)
    plt.plot(dates, equal_portfolio_values, label=f'Equal-Weight Portfolio (Return: {equal_total_return:.2%})', linewidth=2, linestyle='--')
    
    plt.title('Portfolio Backtest Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/portfolio_backtest.png")
    plt.close()
    
    logger.info(f"Backtest results: Recommendation-based return: {rec_total_return:.2%}, Equal-weight return: {equal_total_return:.2%}")
    
    # Save backtest results to JSON
    with open("output/backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """Run the AlphaQuant Demo with real data."""
    logger.info("Starting AlphaQuant Demo (Real Data Version)")
    
    # Initialize data manager
    data_manager = DataManager(cache_dir="data/cache", output_dir="output")
    
    # Define tickers to analyze
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "JNJ", "PG", "V"]
    
    # 1. Fetch and process market data
    logger.info("Fetching and processing market data")
    stock_data = {}
    for ticker in tickers:
        try:
            logger.info(f"Processing {ticker}")
            data = data_manager.get_stock_data(ticker, period="1y")
            stock_data[ticker] = data
            
            # Save a preview of the data
            if not data.empty:
                data.head().to_csv(f"output/{ticker}_preview.csv")
                logger.info(f"Saved preview for {ticker}")
                
                # Plot the stock data
                plt.figure(figsize=(12, 6))
                plt.plot(data.index, data['Close'], label='Close Price')
                plt.plot(data.index, data['SMA_20'], label='20-day SMA', linestyle='--')
                plt.plot(data.index, data['SMA_50'], label='50-day SMA', linestyle='-.')
                plt.title(f"{ticker} Stock Price and Moving Averages")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.legend()
                plt.grid(True)
                plt.savefig(f"output/{ticker}_price_chart.png")
                plt.close()
                logger.info(f"Saved price chart for {ticker}")
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
    
    # 2. Generate market overview with real news
    try:
        logger.info("Generating market overview with real news")
        overview = data_manager.get_market_sentiment()
        
        if "error" not in overview:
            logger.info(f"Market sentiment: {overview['market_sentiment']['market_sentiment_label']}")
            
            # Log some of the headlines
            if "top_headlines" in overview["market_sentiment"]:
                logger.info("Top headlines:")
                for headline in overview["market_sentiment"]["top_headlines"][:5]:
                    logger.info(f"- {headline['title']} (Sentiment: {headline['sentiment']:.2f})")
            
            # Save market overview to JSON
            with open("output/market_overview.json", "w") as f:
                json.dump(overview, f, indent=2)
            logger.info("Saved market overview to output/market_overview.json")
        else:
            logger.error(f"Error in market sentiment: {overview['error']}")
    except Exception as e:
        logger.error(f"Error generating market overview: {e}")
    
    # 3. Generate market map with forecasts
    try:
        logger.info("Generating market map with forecasts")
        data_manager.generate_market_map(tickers, output_path="output/market_map.png")
        logger.info("Market map generated")
    except Exception as e:
        logger.error(f"Error generating market map: {e}")
    
    # 4. Bayesian forecasting
    try:
        logger.info("Running Bayesian forecasting for AAPL")
        forecaster = BayesianForecaster()
        
        if "AAPL" in stock_data and not stock_data["AAPL"].empty:
            forecast_result = forecaster.forecast(stock_data["AAPL"], "AAPL", days=5)
            logger.info("Generating forecast plot")
            forecaster.plot_forecast(stock_data["AAPL"], save_path="output/aapl_forecast.png")
            
            # Log forecast details
            if 'forecast' in forecast_result:
                forecast = forecast_result['forecast']
                logger.info(f"AAPL Forecast for next 5 days: {forecast['forecast_mean']}")
                
                # Save forecast to JSON
                with open("output/aapl_forecast.json", "w") as f:
                    json.dump(forecast, f, indent=2)
                logger.info("Saved AAPL forecast to output/aapl_forecast.json")
    except Exception as e:
        logger.error(f"Error in Bayesian forecasting: {e}")
    
    # 5. Stock recommendations
    try:
        logger.info("Generating stock recommendations")
        all_recommendations = {}
        
        for ticker in tickers:
            if ticker in stock_data and not stock_data[ticker].empty:
                recommendation = data_manager.generate_stock_recommendation(ticker, stock_data[ticker])
                all_recommendations[ticker] = recommendation
                logger.info(f"{ticker} Recommendation: {recommendation['recommendation']} (Confidence: {recommendation['confidence']:.2f})")
                if 'reasons' in recommendation:
                    logger.info(f"Reasons: {recommendation['reasons']}")
        
        # Save all recommendations to JSON
        with open("output/stock_recommendations.json", "w") as f:
            json.dump(all_recommendations, f, indent=2)
        logger.info("Saved stock recommendations to output/stock_recommendations.json")
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")

    # 5.5. Run portfolio backtest
    try:
        logger.info("Running portfolio backtest comparison")
        backtest_results = run_portfolio_backtest(stock_data, tickers, lookback_period=120)
        
        if "error" not in backtest_results:
            recommended_return = backtest_results["recommended_portfolio"]["total_return"]
            equal_return = backtest_results["equal_weight_portfolio"]["total_return"]
            
            logger.info(f"Backtest completed. Recommendation-based vs Equal-weight: {recommended_return:.2%} vs {equal_return:.2%}")
            
            # Calculate outperformance
            outperformance = recommended_return - equal_return
            if outperformance > 0:
                logger.info(f"Recommendation strategy outperformed by {outperformance:.2%}")
            else:
                logger.info(f"Equal-weight strategy outperformed by {-outperformance:.2%}")
                
        else:
            logger.error(f"Backtest error: {backtest_results['error']}")
    except Exception as e:
        logger.error(f"Error in portfolio backtest: {e}")
    
    # 6. Portfolio simulation
    try:
        logger.info("Running portfolio simulation")
        
        # # Create a portfolio with real stocks
        # portfolio_tickers = ["AAPL", "MSFT", "GOOGL"]
        # weights = [1/3, 1/3, 1/3]  # Equal weights

        # Select portfolio based on recommendations
        portfolio_tickers, weights = select_portfolio_based_on_recommendations(
            all_recommendations, tickers, min_confidence=0.6
        )
        
        logger.info(f"Recommendation-based portfolio: {portfolio_tickers}")
        logger.info(f"Portfolio weights: {[round(w, 2) for w in weights]}")
        
        # Calculate portfolio performance
        portfolio_values = []
        dates = []
        
        # Find common dates across all stocks
        common_dates = None
        for ticker in portfolio_tickers:
            if ticker in stock_data and not stock_data[ticker].empty:
                if common_dates is None:
                    common_dates = set(stock_data[ticker].index)
                else:
                    common_dates = common_dates.intersection(set(stock_data[ticker].index))
        
        if common_dates:
            common_dates = sorted(list(common_dates))
            
            # Normalize prices to start at 100
            normalized_prices = {}
            for ticker in portfolio_tickers:
                if ticker in stock_data and not stock_data[ticker].empty:
                    prices = stock_data[ticker]['Close']
                    first_price = prices[prices.index == common_dates[0]].iloc[0]
                    normalized_prices[ticker] = prices / first_price * 100
            
            # Calculate portfolio value
            for date in common_dates:
                portfolio_value = 0
                for i, ticker in enumerate(portfolio_tickers):
                    if ticker in normalized_prices:
                        portfolio_value += weights[i] * normalized_prices[ticker][normalized_prices[ticker].index == date].iloc[0]
                
                portfolio_values.append(portfolio_value)
                dates.append(date)
            
            # Plot portfolio performance
            plt.figure(figsize=(12, 6))
            plt.plot(dates, portfolio_values, label='Portfolio', linewidth=2)
            
            # Plot individual stocks
            for ticker in portfolio_tickers:
                if ticker in normalized_prices:
                    values = [normalized_prices[ticker][normalized_prices[ticker].index == date].iloc[0] for date in dates]
                    plt.plot(dates, values, label=ticker, alpha=0.7)
            
            plt.title('Portfolio Performance')
            plt.xlabel('Date')
            plt.ylabel('Value (normalized)')
            plt.legend()
            plt.grid(True)
            plt.savefig("output/portfolio_performance.png")
            plt.close()
            
            logger.info("Saved portfolio performance chart to output/portfolio_performance.png")
    except Exception as e:
        logger.error(f"Error in portfolio simulation: {e}")
    
    logger.info("AlphaQuant Demo Completed Successfully")
    logger.info("Check the 'output' directory for generated files and visualizations")


if __name__ == "__main__":
    main() 