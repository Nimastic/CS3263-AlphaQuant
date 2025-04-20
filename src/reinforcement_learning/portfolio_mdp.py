import gym
import numpy as np
import pandas as pd
from gym import spaces


class PortfolioMDP(gym.Env):
    """Portfolio allocation as a Markov Decision Process"""
    
    def __init__(self, assets, historical_data, risk_aversion=1.0):
        super(PortfolioMDP, self).__init__()
        self.assets = assets
        self.data = historical_data
        self.risk_aversion = risk_aversion
        
        # Action space: allocation percentage for each asset (plus cash)
        # Each action is a vector of percentages that must sum to 1
        self.action_space = spaces.Box(
            low=0, high=1, shape=(len(assets) + 1,), dtype=np.float32)
        
        # State space: prices, technical indicators, portfolio allocation
        n_features = len(assets) * 3 + (len(assets) + 1)  # price, rsi, vol + current allocation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32)
        
        self.reset()
    
    def reset(self):
        """Reset the environment to starting state"""
        self.current_step = 0
        self.portfolio_value = 10000.0  # Initial capital
        # Equal initial allocation to cash
        self.allocation = np.zeros(len(self.assets) + 1)
        self.allocation[-1] = 1.0  # All in cash
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one time step within the environment"""
        # Normalize action to ensure it sums to 1
        action = action / np.sum(action)
        
        # Calculate portfolio value before reallocation
        prev_value = self.portfolio_value
        
        # Apply transaction costs for reallocation
        costs = self._calculate_transaction_costs(action)
        self.portfolio_value -= costs
        
        # Update allocation
        self.allocation = action
        
        # Move to next time step
        self.current_step += 1
        
        # Update portfolio value based on asset returns
        self._update_portfolio_value()
        
        # Calculate reward (Sharpe ratio component)
        return_pct = (self.portfolio_value - prev_value) / prev_value
        vol = self._portfolio_volatility(window=20)
        reward = return_pct - (self.risk_aversion * vol)
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        """Construct the state observation"""
        if self.current_step >= len(self.data):
            # If we're at the end of data, use the last available data
            step = len(self.data) - 1
        else:
            step = self.current_step
            
        # Get price data and technical indicators for current step
        obs = []
        
        # Add price features
        for asset in self.assets:
            if asset in self.data.columns:
                # Add normalized price
                price = self.data[asset].iloc[step]
                obs.append(price / self.data[asset].iloc[max(0, step-20):step+1].mean())
                
                # Add RSI (if available)
                rsi_col = f"{asset}_RSI"
                if rsi_col in self.data.columns:
                    obs.append(self.data[rsi_col].iloc[step] / 100.0)  # Normalize to 0-1
                else:
                    obs.append(0.5)  # Default value
                
                # Add volatility (if available)
                vol_col = f"{asset}_VOL"
                if vol_col in self.data.columns:
                    obs.append(self.data[vol_col].iloc[step])
                else:
                    obs.append(0.1)  # Default value
            else:
                # If asset data not available, use placeholder values
                obs.extend([1.0, 0.5, 0.1])
        
        # Add current allocation
        obs.extend(self.allocation)
        
        return np.array(obs)
    
    def _update_portfolio_value(self):
        """Update portfolio value based on asset returns"""
        if self.current_step < 1 or self.current_step >= len(self.data):
            return
        
        # Calculate returns for each asset
        returns = np.zeros(len(self.assets) + 1)
        for i, asset in enumerate(self.assets):
            if asset in self.data.columns:
                prev_price = self.data[asset].iloc[self.current_step - 1]
                curr_price = self.data[asset].iloc[self.current_step]
                returns[i] = (curr_price / prev_price) - 1
        
        # Cash return is 0 (or could be set to risk-free rate)
        returns[-1] = 0
        
        # Update portfolio value
        self.portfolio_value *= (1 + np.sum(self.allocation * returns))
    
    def _calculate_transaction_costs(self, new_allocation):
        """Calculate transaction costs for reallocation"""
        # Simple model: 0.1% of traded value
        cost_rate = 0.001
        traded_value = np.sum(np.abs(new_allocation - self.allocation)) * self.portfolio_value / 2
        return traded_value * cost_rate
    
    def _portfolio_volatility(self, window=20):
        """Calculate portfolio volatility based on historical data"""
        if self.current_step < window:
            return 0.01  # Default value for initial steps
        
        # Get historical returns for each asset
        asset_returns = np.zeros((window, len(self.assets) + 1))
        for i, asset in enumerate(self.assets):
            if asset in self.data.columns:
                prices = self.data[asset].iloc[self.current_step - window:self.current_step]
                asset_returns[:, i] = prices.pct_change().dropna().values
        
        # Calculate portfolio return series
        portfolio_returns = np.sum(self.allocation[:-1] * asset_returns[:, :-1], axis=1)
        
        # Return annualized volatility
        return np.std(portfolio_returns) * np.sqrt(252) 