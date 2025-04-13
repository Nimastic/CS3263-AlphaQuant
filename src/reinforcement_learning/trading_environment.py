"""
Trading Environment Module.

This module implements a custom OpenAI Gym environment for simulating
stock trading for reinforcement learning.
"""

import gym
import numpy as np
import pandas as pd
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """
    OpenAI Gym environment for stock trading using reinforcement learning.
    
    This environment simulates stock trading with features like multiple stocks,
    transaction costs, and flexible reward functions.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        tickers: List[str],
        initial_balance: float = 10000.0,
        transaction_cost_pct: float = 0.001,
        window_size: int = 10,
        reward_function: str = 'sharpe',
        max_steps: Optional[int] = None,
        tech_indicators: bool = True,
        sentiment_features: bool = True,
        risk_aversion: float = 1.0
    ):
        """
        Initialize the trading environment.

        Args:
            df: DataFrame with market data for multiple stocks
            tickers: List of stock symbols
            initial_balance: Initial cash balance
            transaction_cost_pct: Transaction cost as percentage
            window_size: Number of time steps to include in state
            reward_function: Type of reward function ('sharpe', 'returns', 'risk_adjusted')
            max_steps: Maximum number of steps per episode (None for full dataset)
            tech_indicators: Whether to include technical indicators in state
            sentiment_features: Whether to include sentiment features in state
            risk_aversion: Risk aversion parameter for reward calculation
        """
        super(TradingEnvironment, self).__init__()

        self.df = df
        self.tickers = tickers
        self.num_stocks = len(tickers)
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.window_size = window_size
        self.reward_function = reward_function
        self.risk_aversion = risk_aversion
        
        # Data preprocessing
        self.close_columns = [f"{ticker}_close" for ticker in tickers]
        self.processed_data = self._preprocess_data(df, tech_indicators, sentiment_features)
        
        # Set up environment parameters
        self.data_length = len(self.processed_data)
        if max_steps is None:
            self.max_steps = self.data_length - window_size - 1
        else:
            self.max_steps = min(max_steps, self.data_length - window_size - 1)
            
        self.current_step = 0
        self.current_episode = 0
        
        # Feature dimension
        self.feature_dim = self.processed_data.shape[1]
        
        # Calculate state dimension
        self.state_dim = (
            self.window_size * self.feature_dim +  # Market data features
            self.num_stocks +  # Portfolio positions
            1  # Cash balance
        )
        
        # Action space: portfolio weights for each asset (including cash)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.num_stocks + 1,), dtype=np.float32
        )
        
        # Observation space: historical market data + portfolio state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Initialize portfolio state
        self.reset()
        
        logger.info(f"TradingEnvironment initialized with {self.num_stocks} stocks and {self.data_length} time steps")

    def _preprocess_data(
        self, 
        df: pd.DataFrame, 
        tech_indicators: bool, 
        sentiment_features: bool
    ) -> np.ndarray:
        """
        Preprocess market data for the environment.

        Args:
            df: DataFrame with market data
            tech_indicators: Whether to include technical indicators
            sentiment_features: Whether to include sentiment features

        Returns:
            Numpy array with preprocessed data
        """
        # Make sure all required columns exist
        for ticker in self.tickers:
            required_cols = [f"{ticker}_close", f"{ticker}_open", f"{ticker}_high", f"{ticker}_low", f"{ticker}_volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns for {ticker}: {missing_cols}")
                
        # Select features
        selected_features = []
        
        # Basic price and volume data
        for ticker in self.tickers:
            selected_features.extend([
                f"{ticker}_close", f"{ticker}_open", f"{ticker}_high", f"{ticker}_low", f"{ticker}_volume"
            ])
            
        # Technical indicators
        if tech_indicators:
            for ticker in self.tickers:
                tech_cols = [col for col in df.columns if ticker in col and any(
                    indicator in col for indicator in [
                        "sma", "ema", "rsi", "macd", "bollinger", "atr", "return", "volatility"
                    ]
                )]
                selected_features.extend(tech_cols)
                
        # Sentiment features
        if sentiment_features:
            sentiment_cols = [col for col in df.columns if "sentiment" in col.lower()]
            selected_features.extend(sentiment_cols)
            
        # Select only columns that exist in the DataFrame
        selected_features = [col for col in selected_features if col in df.columns]
        
        # Scale the data (normalize each feature)
        df_selected = df[selected_features]
        df_scaled = (df_selected - df_selected.mean()) / (df_selected.std() + 1e-10)
        
        # Replace NaN values with 0
        df_scaled = df_scaled.fillna(0)
        
        # Convert to numpy array
        preprocessed_data = df_scaled.values
        
        logger.info(f"Preprocessed data with {len(selected_features)} features")
        return preprocessed_data

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation (state).

        Returns:
            Numpy array with the current state
        """
        # Get market data for the current window
        market_data = self.processed_data[self.current_step:self.current_step + self.window_size]
        market_data_flat = market_data.flatten()
        
        # Get portfolio state
        portfolio_state = np.concatenate([
            self.shares_held,
            [self.balance]
        ])
        
        # Combine market data and portfolio state
        observation = np.concatenate([market_data_flat, portfolio_state])
        
        return observation

    def _calculate_portfolio_value(self) -> float:
        """
        Calculate the current portfolio value.

        Returns:
            Current portfolio value
        """
        # Get current prices
        current_prices = self.processed_data[self.current_step + self.window_size - 1, :self.num_stocks]
        
        # Calculate stock values
        stock_values = self.shares_held * current_prices
        
        # Total portfolio value
        portfolio_value = self.balance + np.sum(stock_values)
        
        return portfolio_value

    def _get_portfolio_weights(self) -> np.ndarray:
        """
        Calculate the current portfolio weights.

        Returns:
            Array of portfolio weights (including cash)
        """
        # Get current prices
        current_prices = self.processed_data[self.current_step + self.window_size - 1, :self.num_stocks]
        
        # Calculate stock values
        stock_values = self.shares_held * current_prices
        
        # Total portfolio value
        portfolio_value = self.balance + np.sum(stock_values)
        
        if portfolio_value <= 0:
            # Default to cash if portfolio value is non-positive
            return np.array([0] * self.num_stocks + [1])
            
        # Calculate weights
        stock_weights = stock_values / portfolio_value
        cash_weight = self.balance / portfolio_value
        
        return np.concatenate([stock_weights, [cash_weight]])

    def _calculate_reward(self) -> float:
        """
        Calculate the reward for the current step.

        Returns:
            Reward value
        """
        # Get current portfolio value
        current_value = self._calculate_portfolio_value()
        
        # Calculate return
        portfolio_return = (current_value - self.previous_value) / self.previous_value
        
        if self.reward_function == 'returns':
            # Simple returns
            reward = portfolio_return
        elif self.reward_function == 'sharpe':
            # Sharpe ratio approximation (assuming daily data)
            self.returns_history.append(portfolio_return)
            if len(self.returns_history) > 1:
                returns_std = np.std(self.returns_history) + 1e-10
                returns_mean = np.mean(self.returns_history)
                reward = returns_mean / returns_std
            else:
                reward = 0
        elif self.reward_function == 'risk_adjusted':
            # Risk-adjusted returns
            self.returns_history.append(portfolio_return)
            if len(self.returns_history) > 1:
                returns_std = np.std(self.returns_history) + 1e-10
                reward = portfolio_return - self.risk_aversion * returns_std
            else:
                reward = portfolio_return
        else:
            # Default to simple returns
            reward = portfolio_return
            
        # Update previous value
        self.previous_value = current_value
        
        return reward

    def _apply_transaction_costs(self, old_weights: np.ndarray, new_weights: np.ndarray) -> float:
        """
        Calculate and apply transaction costs.

        Args:
            old_weights: Previous portfolio weights
            new_weights: New portfolio weights

        Returns:
            Transaction cost amount
        """
        # Calculate turnover
        turnover = np.sum(np.abs(new_weights[:-1] - old_weights[:-1]))
        
        # Calculate cost
        cost = turnover * self.transaction_cost_pct * self._calculate_portfolio_value()
        
        # Apply cost
        self.balance -= cost
        
        return cost

    def _rebalance_portfolio(self, action: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Rebalance the portfolio according to the given action.

        Args:
            action: Target portfolio weights

        Returns:
            Tuple of (transaction_cost, new_weights)
        """
        # Normalize action to ensure it sums to 1
        action = action / (np.sum(action) + 1e-10)
        
        # Get current weights
        old_weights = self._get_portfolio_weights()
        
        # Get current portfolio value
        portfolio_value = self._calculate_portfolio_value()
        
        # Calculate transaction costs
        transaction_cost = self._apply_transaction_costs(old_weights, action)
        
        # Update portfolio value after costs
        portfolio_value -= transaction_cost
        
        # Get current prices
        current_prices = self.processed_data[self.current_step + self.window_size - 1, :self.num_stocks]
        
        # Calculate new share amounts
        new_stock_values = action[:-1] * portfolio_value
        new_shares = new_stock_values / (current_prices + 1e-10)
        
        # Update balance
        new_cash = action[-1] * portfolio_value
        
        # Update portfolio state
        self.shares_held = new_shares
        self.balance = new_cash
        
        # Return cost and new weights
        return transaction_cost, action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: Portfolio allocation weights

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Ensure action is valid
        action = np.clip(action, 0, 1)
        
        # Rebalance portfolio
        cost, new_weights = self._rebalance_portfolio(action)
        
        # Move to next step
        self.current_step += 1
        self.steps_beyond_done = None
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Get next observation
        next_observation = self._get_observation()
        
        # Additional info
        info = {
            'portfolio_value': self._calculate_portfolio_value(),
            'transaction_cost': cost,
            'portfolio_weights': new_weights,
            'step': self.current_step,
            'episode': self.current_episode
        }
        
        return next_observation, reward, done, info

    def reset(self) -> np.ndarray:
        """
        Reset the environment to the initial state.

        Returns:
            Initial observation
        """
        self.current_episode += 1
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = np.zeros(self.num_stocks)
        self.returns_history = []
        self.previous_value = self.initial_balance
        self.steps_beyond_done = None
        
        return self._get_observation()

    def render(self, mode='human'):
        """
        Render the environment.

        Args:
            mode: Rendering mode
        """
        if mode != 'human':
            raise NotImplementedError(f"Render mode {mode} not implemented")
            
        # Get current portfolio value and weights
        portfolio_value = self._calculate_portfolio_value()
        weights = self._get_portfolio_weights()
        
        # Print information about the current state
        print(f"\nEpisode: {self.current_episode}, Step: {self.current_step}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Cash: ${self.balance:.2f} ({weights[-1]*100:.1f}%)")
        
        # Print stock allocations
        print("\nStock Allocations:")
        for i, ticker in enumerate(self.tickers):
            stock_value = self.shares_held[i] * self.processed_data[self.current_step + self.window_size - 1, i]
            print(f"{ticker}: {self.shares_held[i]:.2f} shares, ${stock_value:.2f} ({weights[i]*100:.1f}%)")
            
        print("\n")

    def get_data_window(self, start_idx: int, end_idx: int) -> pd.DataFrame:
        """
        Get a window of the processed data.

        Args:
            start_idx: Start index
            end_idx: End index

        Returns:
            DataFrame with the requested window of data
        """
        if start_idx < 0 or end_idx >= self.data_length:
            raise ValueError("Requested indices are out of bounds")
            
        return pd.DataFrame(
            self.processed_data[start_idx:end_idx],
            columns=[f"feature_{i}" for i in range(self.feature_dim)]
        ) 