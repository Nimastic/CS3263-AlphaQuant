"""
News Sentiment Analyzer Module.

This module processes financial news articles and performs sentiment analysis
to gauge market sentiment for stocks and the overall market.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv
import re
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """Class for analyzing sentiment from financial news sources."""

    def __init__(self, use_news_api: bool = True):
        """
        Initialize the news sentiment analyzer.

        Args:
            use_news_api: Whether to use News API for retrieving articles
        """
        self.use_news_api = use_news_api
        
        if use_news_api:
            self.news_api_key = os.getenv("NEWS_API_KEY")
            if not self.news_api_key:
                logger.warning("NEWS_API_KEY not found in environment variables")
                self.use_news_api = False
        
        logger.info(f"NewsSentimentAnalyzer initialized (News API: {use_news_api})")

    def get_company_news(
        self, 
        company_name: str, 
        ticker: str, 
        days_back: int = 7,
        max_articles: int = 10
    ) -> pd.DataFrame:
        """
        Retrieve news articles for a specific company.

        Args:
            company_name: Name of the company
            ticker: Stock symbol
            days_back: Number of days to look back
            max_articles: Maximum number of articles to retrieve

        Returns:
            DataFrame with news articles and metadata
        """
        if not self.use_news_api:
            return pd.DataFrame()
            
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        # Query terms
        query = f'"{company_name}" OR "{ticker}"'
        
        try:
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': query,
                'from': from_date,
                'to': to_date,
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': max_articles,
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get('status') != 'ok':
                logger.warning(f"Failed to get news for {company_name} ({ticker}): {data.get('message', 'Unknown error')}")
                return pd.DataFrame()
                
            articles = data.get('articles', [])
            
            if not articles:
                logger.info(f"No news articles found for {company_name} ({ticker})")
                return pd.DataFrame()
                
            # Process articles
            processed_articles = []
            for article in articles:
                # Calculate sentiment (simple implementation)
                sentiment_score = self._calculate_simple_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
                
                processed_articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'sentiment_score': sentiment_score
                })
                
            return pd.DataFrame(processed_articles)
            
        except Exception as e:
            logger.warning(f"Error retrieving news for {company_name} ({ticker}): {e}")
            return pd.DataFrame()

    def get_market_news(
        self, 
        days_back: int = 3,
        max_articles: int = 20
    ) -> pd.DataFrame:
        """
        Retrieve general market news.

        Args:
            days_back: Number of days to look back
            max_articles: Maximum number of articles to retrieve

        Returns:
            DataFrame with news articles and metadata
        """
        if not self.use_news_api:
            return pd.DataFrame()
            
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        # Market-related terms
        query = 'stock market OR financial markets OR wall street OR S&P 500 OR nasdaq OR dow jones'
        
        try:
            url = 'https://newsapi.org/v2/everything'
            params = {
                'q': query,
                'from': from_date,
                'to': to_date,
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': max_articles,
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get('status') != 'ok':
                logger.warning(f"Failed to get market news: {data.get('message', 'Unknown error')}")
                return pd.DataFrame()
                
            articles = data.get('articles', [])
            
            if not articles:
                logger.info("No market news articles found")
                return pd.DataFrame()
                
            # Process articles
            processed_articles = []
            for article in articles:
                # Calculate sentiment
                sentiment_score = self._calculate_simple_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
                
                processed_articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'sentiment_score': sentiment_score
                })
                
            return pd.DataFrame(processed_articles)
            
        except Exception as e:
            logger.warning(f"Error retrieving market news: {e}")
            return pd.DataFrame()

    def get_sentiment_summary(self, ticker: str, company_name: str) -> Dict:
        """
        Get a summary of sentiment for a specific company.

        Args:
            ticker: Stock symbol
            company_name: Name of the company

        Returns:
            Dictionary with sentiment summary
        """
        try:
            # Get news articles
            news_df = self.get_company_news(company_name, ticker, days_back=14)
            
            if news_df.empty:
                return {
                    'ticker': ticker,
                    'company_name': company_name,
                    'sentiment_score': 0.0,  # Neutral
                    'sentiment_label': 'neutral',
                    'articles_count': 0,
                    'latest_headlines': []
                }
                
            # Calculate average sentiment
            avg_sentiment = news_df['sentiment_score'].mean()
            
            # Determine sentiment label
            sentiment_label = 'neutral'
            if avg_sentiment >= 0.2:
                sentiment_label = 'positive'
            elif avg_sentiment <= -0.2:
                sentiment_label = 'negative'
                
            # Get latest headlines
            latest_headlines = []
            for _, row in news_df.head(3).iterrows():
                latest_headlines.append({
                    'title': row['title'],
                    'source': row['source'],
                    'sentiment': row['sentiment_score']
                })
                
            return {
                'ticker': ticker,
                'company_name': company_name,
                'sentiment_score': avg_sentiment,
                'sentiment_label': sentiment_label,
                'articles_count': len(news_df),
                'latest_headlines': latest_headlines
            }
            
        except Exception as e:
            logger.warning(f"Error getting sentiment summary for {ticker}: {e}")
            return {
                'ticker': ticker,
                'company_name': company_name,
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'articles_count': 0,
                'latest_headlines': []
            }

    def get_market_sentiment(self) -> Dict:
        """
        Get overall market sentiment.

        Returns:
            Dictionary with market sentiment information
        """
        try:
            # Get market news
            news_df = self.get_market_news(days_back=5, max_articles=30)
            
            if news_df.empty:
                return {
                    'market_sentiment_score': 0.0,
                    'market_sentiment_label': 'neutral',
                    'articles_count': 0,
                    'top_headlines': []
                }
                
            # Calculate average sentiment
            avg_sentiment = news_df['sentiment_score'].mean()
            
            # Determine sentiment label
            sentiment_label = 'neutral'
            if avg_sentiment >= 0.15:
                sentiment_label = 'positive'
            elif avg_sentiment <= -0.15:
                sentiment_label = 'negative'
                
            # Get top headlines
            top_headlines = []
            for _, row in news_df.head(5).iterrows():
                top_headlines.append({
                    'title': row['title'],
                    'source': row['source'],
                    'sentiment': row['sentiment_score']
                })
                
            return {
                'market_sentiment_score': avg_sentiment,
                'market_sentiment_label': sentiment_label,
                'articles_count': len(news_df),
                'top_headlines': top_headlines
            }
            
        except Exception as e:
            logger.warning(f"Error getting market sentiment: {e}")
            return {
                'market_sentiment_score': 0.0,
                'market_sentiment_label': 'neutral',
                'articles_count': 0,
                'top_headlines': []
            }

    def _calculate_simple_sentiment(self, text: str) -> float:
        """
        Calculate a simple sentiment score for text.
        
        This is a very basic implementation that should be replaced with a more sophisticated
        approach like VADER sentiment analysis or a pre-trained model for production use.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        # Simple sentiment word lists
        positive_words = [
            'up', 'rise', 'rising', 'rose', 'gain', 'gains', 'positive', 'improved', 'improving',
            'growth', 'growing', 'grew', 'bullish', 'strong', 'stronger', 'strongest',
            'high', 'higher', 'highest', 'outperform', 'outperforming', 'outperformed',
            'beat', 'beats', 'beating', 'exceeded', 'exceeding', 'exceed', 'successful',
            'success', 'profit', 'profitable', 'rally', 'boom', 'confidence', 'confident',
            'optimistic', 'optimism', 'opportunity', 'opportunities', 'promising'
        ]
        
        negative_words = [
            'down', 'fall', 'falling', 'fell', 'drop', 'drops', 'dropping', 'dropped',
            'decline', 'declines', 'declining', 'declined', 'bearish', 'weak', 'weaker',
            'weakest', 'low', 'lower', 'lowest', 'underperform', 'underperforming',
            'underperformed', 'miss', 'missed', 'missing', 'fail', 'fails', 'failing',
            'failed', 'disappointment', 'disappointing', 'disappointed', 'loss', 'losses',
            'negative', 'trouble', 'struggling', 'struggled', 'struggle', 'crisis',
            'pessimistic', 'pessimism', 'concern', 'concerns', 'concerning', 'worried',
            'worry', 'worries', 'worrying', 'anxious', 'anxiety', 'fear', 'fears'
        ]
        
        # Clean text - lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        
        # Count occurrences
        positive_count = 0
        negative_count = 0
        
        for word in words:
            if word in positive_words:
                positive_count += 1
            elif word in negative_words:
                negative_count += 1
        
        # Calculate sentiment score
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0  # Neutral
            
        return (positive_count - negative_count) / total_sentiment_words

    def scrape_article_content(self, url: str) -> str:
        """
        Scrape the content of an article.

        Args:
            url: URL of the article

        Returns:
            Extracted article text
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            # Get text
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up text (remove extra whitespace)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except RequestException as e:
            logger.warning(f"Error scraping article content from {url}: {e}")
            return "" 