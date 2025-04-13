"""
Bayesian Forecaster Module.

This module implements Bayesian time series forecasting for stock prices and returns,
providing probabilistic predictions with uncertainty estimates.
"""

import pandas as pd
import numpy as np
import pymc as pm
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple
import arviz as az
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BayesianForecaster:
    """
    Bayesian time series forecasting for financial markets.
    
    This class implements Bayesian probabilistic forecasting models for stock prices
    and returns, providing distributions of predicted values rather than point estimates.
    """

    def __init__(self, samples: int = 1000, tune: int = 1000, chains: int = 2, cores: int = 2):
        """
        Initialize the Bayesian forecaster.

        Args:
            samples: Number of samples to draw
            tune: Number of tuning steps
            chains: Number of chains
            cores: Number of cores to use for sampling
        """
        self.samples = samples
        self.tune = tune
        self.chains = chains
        self.cores = cores
        self.model = None
        self.trace = None
        self.forecast_data = None
        self.forecast_samples = None
        self.ticker = None
        
        logger.info(f"BayesianForecaster initialized (samples: {samples}, chains: {chains})")

    def fit_returns_model(self, returns: np.ndarray, window_size: int = 20) -> bool:
        """
        Fit a Bayesian model to returns data.

        Args:
            returns: Array of return values
            window_size: Number of days to consider for autoregression

        Returns:
            Whether model fitting was successful
        """
        try:
            logger.info(f"Fitting Bayesian model to returns data (window_size: {window_size})")
            
            # Create PyMC model
            with pm.Model() as self.model:
                # Priors for auto-regressive coefficients
                beta = pm.Normal('beta', mu=0, sigma=0.5, shape=window_size)
                
                # Prior for intercept
                intercept = pm.Normal('intercept', mu=0, sigma=0.01)
                
                # Prior for observation noise
                sigma = pm.HalfNormal('sigma', sigma=0.1)
                
                # Auto-regressive component
                ar_component = intercept
                for i in range(window_size):
                    if i + window_size < len(returns):
                        ar_component = ar_component + beta[i] * returns[:-window_size + i]
                
                # Expected value
                mu = pm.Deterministic('mu', ar_component)
                
                # Likelihood
                y = pm.Normal('y', mu=mu, sigma=sigma, observed=returns[window_size:])
                
                # Sample from posterior
                self.trace = pm.sample(
                    draws=self.samples, 
                    tune=self.tune, 
                    chains=self.chains, 
                    cores=self.cores,
                    return_inferencedata=True
                )
            
            logger.info("Bayesian model fitted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting Bayesian model: {e}")
            return False

    def fit_price_model(self, prices: np.ndarray, forecast_horizon: int = 5) -> bool:
        """
        Fit a Bayesian model to price data.

        Args:
            prices: Array of price values
            forecast_horizon: Number of days to forecast

        Returns:
            Whether model fitting was successful
        """
        try:
            logger.info(f"Fitting Bayesian model to price data (horizon: {forecast_horizon})")
            
            # Convert to log prices for better modeling
            log_prices = np.log(prices)
            num_points = len(log_prices)
            
            # Time index
            time_index = np.arange(num_points)
            
            # Create PyMC model
            with pm.Model() as self.model:
                # Linear trend component
                intercept = pm.Normal('intercept', mu=0, sigma=5)
                slope = pm.Normal('slope', mu=0, sigma=0.1)
                
                # Noise term
                sigma = pm.HalfNormal('sigma', sigma=0.1)
                
                # Seasonality (if we're modeling daily data with weekly patterns)
                # For simplicity, not included here
                
                # Expected value (trend model)
                mu = intercept + slope * time_index
                
                # Add auto-regressive component (simple AR(1) process)
                ar_coef = pm.Normal('ar_coef', mu=0.95, sigma=0.05)  # Strong prior near 1
                
                # True AR process (combining deterministic trend and AR component)
                true_ar = np.zeros(num_points)
                true_ar[0] = mu[0]
                for i in range(1, num_points):
                    true_ar[i] = mu[i] + ar_coef * (log_prices[i-1] - mu[i-1])
                
                # Likelihood
                y = pm.Normal('y', mu=true_ar, sigma=sigma, observed=log_prices)
                
                # Sample from posterior
                self.trace = pm.sample(
                    draws=self.samples, 
                    tune=self.tune, 
                    chains=self.chains, 
                    cores=self.cores,
                    return_inferencedata=True
                )
                
                # Store data for forecasting
                self.forecast_data = {
                    'log_prices': log_prices,
                    'prices': prices,
                    'time_index': time_index,
                    'forecast_horizon': forecast_horizon
                }
            
            logger.info("Bayesian price model fitted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error fitting Bayesian price model: {e}")
            return False

    def forecast(self, num_days: int = 5, return_samples: bool = False) -> Dict:
        """
        Generate forecasts from the fitted model.

        Args:
            num_days: Number of days to forecast
            return_samples: Whether to return posterior samples

        Returns:
            Dictionary with forecast information
        """
        if self.model is None or self.trace is None:
            logger.error("Model has not been fitted yet")
            return {'error': 'Model not fitted'}
            
        try:
            logger.info(f"Generating {num_days}-day forecast")
            
            # Get model parameters from trace
            posterior_samples = az.extract(self.trace, var_names=['intercept', 'slope', 'ar_coef', 'sigma'])
            intercept_samples = posterior_samples['intercept'].values
            slope_samples = posterior_samples['slope'].values
            ar_coef_samples = posterior_samples['ar_coef'].values
            sigma_samples = posterior_samples['sigma'].values
            
            # Parameters for forecast
            n_samples = len(intercept_samples)
            log_prices = self.forecast_data['log_prices']
            time_index = self.forecast_data['time_index']
            last_time_idx = time_index[-1]
            last_log_price = log_prices[-1]
            
            # Generate forecasts
            forecast_time_idx = np.arange(last_time_idx + 1, last_time_idx + num_days + 1)
            forecasts = np.zeros((n_samples, num_days))
            
            for i in range(n_samples):
                # Initialize with last observed value
                prev_log_price = last_log_price
                
                for j in range(num_days):
                    # Trend component
                    trend = intercept_samples[i] + slope_samples[i] * forecast_time_idx[j]
                    
                    # Add AR component
                    ar_component = ar_coef_samples[i] * (prev_log_price - (intercept_samples[i] + slope_samples[i] * (forecast_time_idx[j] - 1)))
                    
                    # Generate forecast
                    mu = trend + ar_component
                    
                    # Add noise
                    forecasted_log_price = np.random.normal(mu, sigma_samples[i])
                    
                    # Store
                    forecasts[i, j] = forecasted_log_price
                    
                    # Update for next step
                    prev_log_price = forecasted_log_price
            
            # Convert log forecasts back to price space
            price_forecasts = np.exp(forecasts)
            
            # Compute statistics
            mean_forecasts = np.mean(price_forecasts, axis=0)
            lower_95 = np.percentile(price_forecasts, 2.5, axis=0)
            upper_95 = np.percentile(price_forecasts, 97.5, axis=0)
            
            # Store forecast samples
            self.forecast_samples = price_forecasts
            
            # Prepare results
            last_price = self.forecast_data['prices'][-1]
            forecast_dates = []
            
            # Generate dates for forecast
            last_date = datetime.now()
            for i in range(num_days):
                next_date = last_date + timedelta(days=i+1)
                # Skip weekends
                while next_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                    next_date += timedelta(days=1)
                forecast_dates.append(next_date.strftime('%Y-%m-%d'))
            
            # Prepare forecast dictionary
            forecast_dict = {
                'last_price': last_price,
                'forecast_mean': mean_forecasts.tolist(),
                'forecast_lower': lower_95.tolist(),
                'forecast_upper': upper_95.tolist(),
                'dates': forecast_dates,
                'ticker': self.ticker
            }
            
            if return_samples:
                forecast_dict['samples'] = price_forecasts.tolist()
                
            return forecast_dict
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return {'error': str(e)}

    def fit_model_for_ticker(self, df: pd.DataFrame, ticker: str, forecast_days: int = 5) -> Dict:
        """
        Fit a model for a specific ticker using its historical data.

        Args:
            df: DataFrame with historical data
            ticker: Stock symbol
            forecast_days: Number of days to forecast

        Returns:
            Dictionary with model results and forecast
        """
        try:
            self.ticker = ticker
            logger.info(f"Fitting Bayesian model for {ticker}")
            
            # Make sure DataFrame is not empty
            if df.empty:
                return {'error': f'No data available for {ticker}'}
                
            # Get close prices
            if 'close' in df.columns:
                prices = df['close'].values
            else:
                return {'error': 'Close price column not found in DataFrame'}
                
            # Fit model
            success = self.fit_price_model(prices)
            
            if not success:
                return {'error': 'Failed to fit Bayesian model'}
                
            # Generate forecast
            forecast = self.forecast(num_days=forecast_days, return_samples=False)
            
            # Add model diagnostics
            summary = az.summary(self.trace)
            
            # Convert summary to dict
            summary_dict = summary.to_dict()
            
            return {
                'ticker': ticker,
                'forecast': forecast,
                'model_summary': summary_dict,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fitting model for {ticker}: {e}")
            return {'error': str(e)}

    def plot_forecast(self, save_path: Optional[str] = None) -> bool:
        """
        Plot the forecasted values with uncertainty intervals.

        Args:
            save_path: Path to save the plot (if None, displays the plot)

        Returns:
            Whether plotting was successful
        """
        try:
            if self.forecast_samples is None:
                logger.error("No forecast available to plot")
                return False
                
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Get historical data
            historical_prices = self.forecast_data['prices']
            
            # Plot historical data
            plt.plot(range(len(historical_prices)), historical_prices, 'b-', label='Historical')
            
            # Get forecast statistics
            n_samples, n_days = self.forecast_samples.shape
            mean_forecast = np.mean(self.forecast_samples, axis=0)
            lower_95 = np.percentile(self.forecast_samples, 2.5, axis=0)
            upper_95 = np.percentile(self.forecast_samples, 97.5, axis=0)
            
            # Create x-axis for forecast
            forecast_x = np.arange(len(historical_prices) - 1, len(historical_prices) + n_days - 1)
            
            # Plot forecast
            plt.plot(forecast_x, np.append(historical_prices[-1], mean_forecast), 'r-', label='Forecast')
            plt.fill_between(
                forecast_x, 
                np.append(historical_prices[-1], lower_95),
                np.append(historical_prices[-1], upper_95),
                color='r', alpha=0.2, label='95% Interval'
            )
            
            # Customize plot
            plt.title(f'Bayesian Forecast for {self.ticker}')
            plt.xlabel('Trading Days')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save or display
            if save_path:
                plt.savefig(save_path)
                plt.close()
                logger.info(f"Forecast plot saved to {save_path}")
            else:
                plt.show()
                
            return True
            
        except Exception as e:
            logger.error(f"Error plotting forecast: {e}")
            return False

    def predict_probability_of_rise(self, threshold: float = 0.0) -> float:
        """
        Calculate the probability that the stock price will rise above a threshold.

        Args:
            threshold: Threshold for percentage increase

        Returns:
            Probability of price increase above threshold
        """
        if self.forecast_samples is None:
            logger.error("No forecast available")
            return 0.0
            
        try:
            # Get the last price
            last_price = self.forecast_data['prices'][-1]
            
            # Calculate returns for the forecast horizon
            n_samples, n_days = self.forecast_samples.shape
            
            # Get forecasted prices at the end of the horizon
            final_prices = self.forecast_samples[:, -1]
            
            # Calculate percentage changes
            percentage_changes = (final_prices - last_price) / last_price
            
            # Calculate probability of rise above threshold
            prob_rise = np.mean(percentage_changes > threshold)
            
            logger.info(f"Probability of {threshold*100:.1f}% rise: {prob_rise:.2f}")
            return prob_rise
            
        except Exception as e:
            logger.error(f"Error calculating probability of rise: {e}")
            return 0.0 