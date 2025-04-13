import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from PIL import Image

# Import the demo script to use its functionality
from demo import DataManager, BayesianForecaster

st.set_page_config(page_title="AlphaQuant Dashboard", layout="wide", page_icon="📊")

# Define the main app
def main():
    # Sidebar for navigation
    st.sidebar.title("AlphaQuant")
    page = st.sidebar.selectbox(
        "Navigation", 
        ["Dashboard", "Market Map", "Stock Analysis", "Forecasting", "Portfolio Simulation", "About"]
    )
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Market Map":
        show_market_map()
    elif page == "Stock Analysis":
        show_stock_analysis()
    elif page == "Forecasting":
        show_forecasting()
    elif page == "Portfolio Simulation":
        show_portfolio_simulation()
    elif page == "About":
        show_about()


def show_dashboard():
    st.title("AlphaQuant Dashboard")
    st.subheader("AI-Powered Investment Advisor")
    
    # Market Overview
    st.markdown("## Market Overview")
    
    try:
        with open('output/market_overview.json', 'r') as f:
            market_overview = json.load(f)
        
        sentiment = market_overview['market_sentiment']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Market Sentiment", 
                sentiment['market_sentiment_label'],
                f"{sentiment['market_sentiment_score']:.2f}"
            )
        
        with col2:
            st.metric("Articles Analyzed", sentiment['articles_count'])
        
        with col3:
            if 'timestamp' in market_overview:
                timestamp = datetime.fromisoformat(market_overview['timestamp'])
                st.metric("Last Updated", timestamp.strftime("%Y-%m-%d %H:%M"))
            else:
                now = datetime.now()
                st.metric("Last Updated", now.strftime("%Y-%m-%d %H:%M"))
        
        # Display top headlines
        if 'top_headlines' in sentiment:
            st.markdown("### Top Headlines")
            for headline in sentiment['top_headlines']:
                sentiment_color = "green" if headline['sentiment'] > 0 else "red"
                
                # Add URL if available
                if 'url' in headline:
                    headline_text = f"[{headline['title']}]({headline['url']})"
                else:
                    headline_text = headline['title']
                
                st.markdown(
                    f"<div style='padding: 10px; border-left: 5px solid {sentiment_color}; margin-bottom: 10px;'>"
                    f"<p>{headline_text}</p>"
                    f"<p><small>Sentiment: {headline['sentiment']:.2f}</small></p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
    except Exception as e:
        st.error(f"Error loading market overview: {e}")
        st.info("Run the demo.py script first to generate the output files")
    
    # Market Map Preview
    st.markdown("## Market Map")
    try:
        if os.path.exists('output/market_map.png'):
            image = Image.open('output/market_map.png')
            st.image(image, use_column_width=True)
            st.caption("Market map showing stock performance vs. volatility with forecasts indicated by arrows")
        else:
            st.info("Market map not found. Run the demo script first.")
    except Exception as e:
        st.error(f"Error loading market map: {e}")
    
    # Stock Recommendations
    st.markdown("## Stock Recommendations")
    
    try:
        with open('output/stock_recommendations.json', 'r') as f:
            recommendations = json.load(f)
        
        # Display recommendations
        cols = st.columns(5)  # Limit to 5 columns per row for better display
        
        for i, (ticker, data) in enumerate(list(recommendations.items())[:10]):  # Show only top 10
            col_idx = i % 5
            with cols[col_idx]:
                rec = data['recommendation'].upper()
                
                # Color based on recommendation
                if rec == "BUY":
                    rec_color = "green"
                elif rec == "SELL":
                    rec_color = "red"
                else:
                    rec_color = "orange"
                
                st.markdown(f"### {ticker}")
                st.markdown(
                    f"<div style='background-color: {rec_color}; padding: 10px; "
                    f"border-radius: 5px; color: white; text-align: center; font-weight: bold;'>"
                    f"{rec}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Confidence:** {data['confidence'] * 100:.1f}%")
                
                # Display first reason only to save space
                if 'reasons' in data and data['reasons']:
                    st.markdown(f"- {data['reasons'][0]}")
    except Exception as e:
        st.error(f"Error loading recommendations: {e}")
    
    # Portfolio Performance
    st.markdown("## Portfolio Performance")
    
    try:
        # Display the portfolio performance chart
        if os.path.exists('output/portfolio_performance.png'):
            image = Image.open('output/portfolio_performance.png')
            st.image(image, use_column_width=True)
        else:
            st.info("Portfolio performance chart not found. Run the demo script first.")
    except Exception as e:
        st.error(f"Error loading portfolio performance: {e}")


def show_market_map():
    st.title("Market Map")
    st.markdown("### Performance vs. Volatility with Price Forecasts")
    
    try:
        if os.path.exists('output/market_map.png'):
            image = Image.open('output/market_map.png')
            st.image(image, use_column_width=True)
            
            st.markdown("""
            ## Understanding the Market Map
            
            This visualization provides a comprehensive view of the market landscape:
            
            - **X-axis**: 30-day price volatility (%)
            - **Y-axis**: Price change over the last 6 months (%)
            - **Bubble size**: Average trading volume (larger = higher volume)
            - **Colors**: Industry sector classification
            - **Arrows**: Price forecasts for the next 30 days
            
            ### Key Insights:
            
            - **High volatility, high return** stocks (upper right) offer growth potential with higher risk
            - **Low volatility, positive return** stocks (upper left) may provide more stable growth
            - **Negative return** stocks below the horizontal line are underperforming
            - **Forecast arrows** indicate our model's price predictions based on Bayesian forecasting
            """)
            
            # Option to generate new market map
            if st.button("Generate New Market Map"):
                with st.spinner("Generating new market map..."):
                    data_manager = DataManager(cache_dir="data/cache", output_dir="output")
                    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "JNJ", "PG", "V"]
                    result = data_manager.generate_market_map(tickers, output_path="output/market_map.png")
                    
                    if result:
                        st.success("Market map updated successfully!")
                        # Refresh the image
                        image = Image.open('output/market_map.png')
                        st.image(image, use_column_width=True)
                    else:
                        st.error("Failed to generate new market map")
        else:
            st.info("Market map not found. Run the demo script first.")
            
            # Option to generate market map if not found
            if st.button("Generate Market Map"):
                with st.spinner("Generating market map..."):
                    data_manager = DataManager(cache_dir="data/cache", output_dir="output")
                    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "JNJ", "PG", "V"]
                    result = data_manager.generate_market_map(tickers, output_path="output/market_map.png")
                    
                    if result:
                        st.success("Market map generated successfully!")
                        # Display the new image
                        image = Image.open('output/market_map.png')
                        st.image(image, use_column_width=True)
                    else:
                        st.error("Failed to generate market map")
    except Exception as e:
        st.error(f"Error loading market map: {e}")


def show_stock_analysis():
    st.title("Stock Analysis")
    
    # Stock selection
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "JNJ", "PG", "V"]
    selected_ticker = st.selectbox("Select a Stock", tickers)
    
    col1, col2 = st.columns([2, 1])
    
    # Display stock chart
    chart_path = f"output/{selected_ticker}_price_chart.png"
    if os.path.exists(chart_path):
        with col1:
            st.markdown(f"### {selected_ticker} Price Chart")
            image = Image.open(chart_path)
            st.image(image, use_column_width=True)
    else:
        with col1:
            st.info(f"Chart for {selected_ticker} not found. Run the demo script first.")
    
    # Display stock data
    preview_path = f"output/{selected_ticker}_preview.csv"
    if os.path.exists(preview_path):
        with col2:
            st.markdown("### Recent Stock Data")
            data = pd.read_csv(preview_path, index_col=0)
            st.dataframe(data)
            
            # Get recommendation
            try:
                with open('output/stock_recommendations.json', 'r') as f:
                    recommendations = json.load(f)
                
                if selected_ticker in recommendations:
                    rec = recommendations[selected_ticker]
                    st.markdown("### Recommendation")
                    
                    # Display recommendation in a box with color
                    rec_text = rec['recommendation'].upper()
                    if rec_text == "BUY":
                        rec_color = "green"
                    elif rec_text == "SELL":
                        rec_color = "red"
                    else:
                        rec_color = "orange"
                    
                    st.markdown(
                        f"<div style='background-color: {rec_color}; padding: 10px; "
                        f"border-radius: 5px; color: white; text-align: center; font-weight: bold;'>"
                        f"{rec_text}</div>",
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(f"**Confidence:** {rec['confidence'] * 100:.1f}%")
                    
                    if 'reasons' in rec:
                        st.markdown("**Reasons:**")
                        for reason in rec['reasons']:
                            st.markdown(f"- {reason}")
            except Exception as e:
                st.error(f"Error loading recommendation: {e}")
    else:
        with col2:
            st.info(f"Data for {selected_ticker} not found. Run the demo script first.")
    
    # Interactive Data Fetching
    st.markdown("## Fetch New Data")
    st.info("This will fetch fresh data from Yahoo Finance")
    
    if st.button("Fetch Latest Data"):
        with st.spinner("Fetching data..."):
            # Use the DataManager to fetch data
            data_manager = DataManager(cache_dir="data/cache", output_dir="output")
            data = data_manager.get_stock_data(selected_ticker, period="1y")
            
            if not data.empty:
                st.success(f"Successfully fetched data for {selected_ticker}")
                st.dataframe(data.tail())
                
                # Plot the data
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(data.index, data['Close'], label='Close')
                ax.plot(data.index, data['SMA_20'], label='20-day SMA', linestyle='--')
                ax.plot(data.index, data['SMA_50'], label='50-day SMA', linestyle='-.')
                ax.set_title(f"{selected_ticker} Stock Price")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            else:
                st.error(f"No data retrieved for {selected_ticker}")


def show_forecasting():
    st.title("Price Forecasting")
    
    # Stock selection for forecasting
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "JNJ", "PG", "V"]
    selected_ticker = st.selectbox("Select a Stock for Forecasting", tickers)
    
    # Show existing forecast if available
    if selected_ticker == "AAPL" and os.path.exists('output/aapl_forecast.png'):
        st.markdown(f"### AAPL Price Forecast")
        image = Image.open('output/aapl_forecast.png')
        st.image(image, use_column_width=True)
        
        # Display forecast details
        try:
            with open('output/aapl_forecast.json', 'r') as f:
                forecast_data = json.load(f)
            
            st.markdown("### Forecast Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Last Price", f"${forecast_data['last_price']:.2f}")
            
            with col2:
                forecast_mean = forecast_data['forecast_mean'][-1]
                change = forecast_mean - forecast_data['last_price']
                change_pct = (change / forecast_data['last_price']) * 100
                st.metric(
                    "Forecasted Price", 
                    f"${forecast_mean:.2f}", 
                    f"{change_pct:.2f}%"
                )
            
            with col3:
                st.metric(
                    "Forecast Period", 
                    f"{forecast_data['forecast_start_date']} to {forecast_data['forecast_end_date']}"
                )
            
            # Create a forecast dataframe for display
            forecast_df = pd.DataFrame({
                'Date': pd.date_range(
                    start=forecast_data['forecast_start_date'], 
                    end=forecast_data['forecast_end_date']
                ),
                'Forecast': forecast_data['forecast_mean'],
                'Lower Bound': forecast_data['forecast_lower_bound'],
                'Upper Bound': forecast_data['forecast_upper_bound']
            })
            
            st.dataframe(forecast_df)
            
        except Exception as e:
            st.error(f"Error loading forecast details: {e}")
    
    # Generate new forecasts
    st.markdown("## Generate New Forecast")
    
    forecast_days = st.slider("Forecast Days", min_value=1, max_value=30, value=5)
    
    if st.button("Generate Forecast"):
        with st.spinner(f"Generating forecast for {selected_ticker}..."):
            # Use the DataManager to fetch data
            data_manager = DataManager(cache_dir="data/cache", output_dir="output")
            data = data_manager.get_stock_data(selected_ticker, period="1y")
            
            if not data.empty:
                # Use BayesianForecaster for forecasting
                forecaster = BayesianForecaster()
                forecast_result = forecaster.forecast(data, selected_ticker, days=forecast_days)
                
                if 'forecast' in forecast_result:
                    st.success(f"Successfully generated forecast for {selected_ticker}")
                    
                    # Create and display forecast plot
                    forecaster.plot_forecast(data, save_path=None)
                    
                    # Get the plot and display it
                    fig = plt.gcf()
                    st.pyplot(fig)
                    
                    # Display forecast details
                    forecast = forecast_result['forecast']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Last Price", f"${forecast['last_price']:.2f}")
                    
                    with col2:
                        forecast_mean = forecast['forecast_mean'][-1]
                        change = forecast_mean - forecast['last_price']
                        change_pct = (change / forecast['last_price']) * 100
                        st.metric(
                            "Forecasted Price", 
                            f"${forecast_mean:.2f}", 
                            f"{change_pct:.2f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "Forecast Period", 
                            f"{forecast['forecast_start_date']} to {forecast['forecast_end_date']}"
                        )
                    
                    # Create a forecast dataframe for display
                    forecast_df = pd.DataFrame({
                        'Date': pd.date_range(
                            start=forecast['forecast_start_date'], 
                            end=forecast['forecast_end_date']
                        ),
                        'Forecast': forecast['forecast_mean'],
                        'Lower Bound': forecast['forecast_lower_bound'],
                        'Upper Bound': forecast['forecast_upper_bound']
                    })
                    
                    st.dataframe(forecast_df)
                else:
                    st.error("Failed to generate forecast")
            else:
                st.error(f"No data retrieved for {selected_ticker}")


def show_portfolio_simulation():
    st.title("Portfolio Simulation")
    
    # Display existing portfolio performance
    if os.path.exists('output/portfolio_performance.png'):
        st.markdown("## Current Portfolio")
        image = Image.open('output/portfolio_performance.png')
        st.image(image, use_column_width=True)
    
    # Create new portfolio simulation
    st.markdown("## Create New Portfolio")
    
    # Stock selection with multiselect
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "JNJ", "PG", "V"]
    selected_tickers = st.multiselect("Select Stocks", tickers, default=["AAPL", "MSFT", "GOOGL"])
    
    # Get weights for each stock
    if selected_tickers:
        st.markdown("### Allocation Weights")
        st.info("Specify the weight for each stock in your portfolio. The weights will be normalized to sum to 100%.")
        
        weights = {}
        cols = st.columns(len(selected_tickers))
        
        for i, ticker in enumerate(selected_tickers):
            with cols[i]:
                weights[ticker] = st.slider(
                    f"{ticker} Weight", 
                    min_value=1, 
                    max_value=100, 
                    value=100 // len(selected_tickers),
                    key=f"weight_{ticker}"
                )
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {ticker: weight / total_weight for ticker, weight in weights.items()}
        
        # Display normalized weights
        st.markdown("### Normalized Weights")
        
        norm_cols = st.columns(len(selected_tickers))
        for i, ticker in enumerate(selected_tickers):
            with norm_cols[i]:
                st.metric(ticker, f"{normalized_weights[ticker] * 100:.1f}%")
        
        # Simulation period
        st.markdown("### Simulation Period")
        period = st.selectbox(
            "Select Period", 
            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            index=3
        )
        
        if st.button("Run Simulation"):
            if len(selected_tickers) > 0:
                with st.spinner("Running portfolio simulation..."):
                    # Use the DataManager to fetch data
                    data_manager = DataManager(cache_dir="data/cache", output_dir="output")
                    
                    # Get data for all selected tickers
                    stock_data = {}
                    for ticker in selected_tickers:
                        data = data_manager.get_stock_data(ticker, period=period)
                        if not data.empty:
                            stock_data[ticker] = data
                    
                    if stock_data:
                        # Find common dates
                        common_dates = None
                        for ticker in selected_tickers:
                            if ticker in stock_data and not stock_data[ticker].empty:
                                if common_dates is None:
                                    common_dates = set(stock_data[ticker].index)
                                else:
                                    common_dates = common_dates.intersection(set(stock_data[ticker].index))
                        
                        if common_dates:
                            common_dates = sorted(list(common_dates))
                            
                            # Normalize prices to start at 100
                            normalized_prices = {}
                            for ticker in selected_tickers:
                                if ticker in stock_data and not stock_data[ticker].empty:
                                    prices = stock_data[ticker]['Close']
                                    first_price = prices[prices.index == common_dates[0]].iloc[0]
                                    normalized_prices[ticker] = prices / first_price * 100
                            
                            # Calculate portfolio value
                            portfolio_values = []
                            dates = []
                            
                            for date in common_dates:
                                portfolio_value = 0
                                for ticker in selected_tickers:
                                    if ticker in normalized_prices:
                                        portfolio_value += normalized_weights[ticker] * normalized_prices[ticker][normalized_prices[ticker].index == date].iloc[0]
                                
                                portfolio_values.append(portfolio_value)
                                dates.append(date)
                            
                            # Plot portfolio performance
                            fig, ax = plt.subplots(figsize=(12, 6))
                            ax.plot(dates, portfolio_values, label='Portfolio', linewidth=2)
                            
                            # Plot individual stocks
                            for ticker in selected_tickers:
                                if ticker in normalized_prices:
                                    values = [normalized_prices[ticker][normalized_prices[ticker].index == date].iloc[0] for date in dates]
                                    ax.plot(dates, values, label=ticker, alpha=0.7)
                            
                            ax.set_title('Portfolio Performance')
                            ax.set_xlabel('Date')
                            ax.set_ylabel('Value (normalized)')
                            ax.legend()
                            ax.grid(True)
                            
                            st.pyplot(fig)
                            
                            # Calculate performance metrics
                            start_value = portfolio_values[0]
                            end_value = portfolio_values[-1]
                            pct_change = (end_value - start_value) / start_value * 100
                            
                            # Display metrics
                            st.markdown("### Performance Metrics")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Return", f"{pct_change:.2f}%")
                            
                            with col2:
                                # Calculate annualized return
                                days = (dates[-1] - dates[0]).days
                                years = days / 365
                                if years > 0:
                                    annualized = ((1 + pct_change / 100) ** (1 / years) - 1) * 100
                                    st.metric("Annualized Return", f"{annualized:.2f}%")
                                else:
                                    st.metric("Annualized Return", "N/A")
                            
                            with col3:
                                # Calculate volatility (std dev of returns)
                                portfolio_returns = [0]
                                for i in range(1, len(portfolio_values)):
                                    daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] * 100
                                    portfolio_returns.append(daily_return)
                                
                                volatility = np.std(portfolio_returns)
                                st.metric("Daily Volatility", f"{volatility:.2f}%")
                        else:
                            st.error("No common dates found for the selected stocks")
                    else:
                        st.error("Failed to fetch data for selected stocks")
            else:
                st.error("Please select at least one stock")


def show_about():
    st.title("About AlphaQuant")
    
    st.markdown("""
    ## AI-Powered Investment Advisor
    
    AlphaQuant is an AI-powered personalized investment advisor that provides data-driven insights, 
    predictions, and portfolio optimization using cutting-edge techniques in machine learning, 
    Bayesian forecasting, and reinforcement learning.
    
    ### Key Features
    
    - **Data Pipeline**: Collect, clean, and process market data from various sources
    - **Sentiment Analysis**: Analyze news and social media for market sentiment
    - **Bayesian Forecasting**: Generate probabilistic price forecasts with uncertainty quantification
    - **Machine Learning Predictions**: Predict market movements using traditional ML and deep learning
    - **Portfolio Optimization**: Optimize portfolios based on user preferences and risk tolerance
    - **Reinforcement Learning**: Develop adaptive trading strategies using RL techniques
    
    ### About This Demo
    
    This Streamlit application demonstrates the capabilities of AlphaQuant in a user-friendly interface.
    The demo uses real-time data from Yahoo Finance and implements simplified versions of the full
    AlphaQuant system's features.
    
    The full version of AlphaQuant includes more advanced models, API endpoints, and a comprehensive
    portfolio management system.
    """)


if __name__ == "__main__":
    main() 