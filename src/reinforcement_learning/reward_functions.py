"""
Reward Functions Module.

This module implements various reward functions for the reinforcement
learning trading environment.
"""

import numpy as np
from typing import List, Optional, Callable
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskAdjustedReturns:
    """Class that provides different risk-adjusted reward functions for RL agents."""

    @staticmethod
    def simple_returns(
        portfolio_value: float,
        previous_value: float
    ) -> float:
        """
        Calculate simple returns as reward.

        Args:
            portfolio_value: Current portfolio value
            previous_value: Previous portfolio value

        Returns:
            Simple return as reward
        """
        return (portfolio_value - previous_value) / previous_value

    @staticmethod
    def sharpe_ratio(
        returns: List[float],
        risk_free_rate: float = 0.0,
        window_size: int = 20
    ) -> float:
        """
        Calculate Sharpe ratio as reward.

        Args:
            returns: List of historical returns
            risk_free_rate: Risk-free rate (daily)
            window_size: Window size for Sharpe calculation

        Returns:
            Sharpe ratio as reward
        """
        if len(returns) < 2:
            return 0.0
            
        # Use the most recent returns up to window_size
        recent_returns = returns[-window_size:]
        
        # Calculate mean and std of returns
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns) + 1e-10  # Add small value to avoid division by zero
        
        # Calculate Sharpe ratio (annualized)
        sharpe = (mean_return - risk_free_rate) / std_return * np.sqrt(252)
        
        return sharpe

    @staticmethod
    def sortino_ratio(
        returns: List[float],
        risk_free_rate: float = 0.0,
        window_size: int = 20
    ) -> float:
        """
        Calculate Sortino ratio as reward.

        Args:
            returns: List of historical returns
            risk_free_rate: Risk-free rate (daily)
            window_size: Window size for Sortino calculation

        Returns:
            Sortino ratio as reward
        """
        if len(returns) < 2:
            return 0.0
            
        # Use the most recent returns up to window_size
        recent_returns = returns[-window_size:]
        
        # Calculate mean return and downside deviation
        mean_return = np.mean(recent_returns)
        
        # Downside returns (only negative returns)
        downside_returns = [r for r in recent_returns if r < 0]
        
        if not downside_returns:
            # No downside returns, use a small value to avoid division by zero
            downside_deviation = 1e-10
        else:
            # Calculate downside deviation
            downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))
            
        # Calculate Sortino ratio (annualized)
        sortino = (mean_return - risk_free_rate) / downside_deviation * np.sqrt(252)
        
        return sortino

    @staticmethod
    def calmar_ratio(
        returns: List[float],
        portfolio_values: List[float],
        window_size: int = 252  # Typically use 1 year of data
    ) -> float:
        """
        Calculate Calmar ratio as reward.

        Args:
            returns: List of historical returns
            portfolio_values: List of historical portfolio values
            window_size: Window size for Calmar calculation

        Returns:
            Calmar ratio as reward
        """
        if len(returns) < window_size or len(portfolio_values) < window_size:
            return 0.0
            
        # Use the most recent data up to window_size
        recent_returns = returns[-window_size:]
        recent_values = portfolio_values[-window_size:]
        
        # Calculate annualized return
        annualized_return = np.mean(recent_returns) * 252
        
        # Calculate maximum drawdown
        max_drawdown = 0.0
        peak_value = recent_values[0]
        
        for value in recent_values:
            if value > peak_value:
                peak_value = value
            drawdown = (peak_value - value) / peak_value
            max_drawdown = max(max_drawdown, drawdown)
            
        # Avoid division by zero
        if max_drawdown < 1e-10:
            max_drawdown = 1e-10
            
        # Calculate Calmar ratio
        calmar = annualized_return / max_drawdown
        
        return calmar

    @staticmethod
    def omega_ratio(
        returns: List[float],
        threshold: float = 0.0,
        window_size: int = 20
    ) -> float:
        """
        Calculate Omega ratio as reward.

        Args:
            returns: List of historical returns
            threshold: Minimum acceptable return
            window_size: Window size for Omega calculation

        Returns:
            Omega ratio as reward
        """
        if len(returns) < 2:
            return 0.0
            
        # Use the most recent returns up to window_size
        recent_returns = returns[-window_size:]
        
        # Calculate gains and losses relative to threshold
        gains = [max(0, r - threshold) for r in recent_returns]
        losses = [max(0, threshold - r) for r in recent_returns]
        
        # Avoid division by zero
        if sum(losses) < 1e-10:
            return 10.0  # Large value to indicate good performance with no losses
            
        # Calculate Omega ratio
        omega = sum(gains) / sum(losses)
        
        return omega

    @staticmethod
    def risk_adjusted_returns(
        portfolio_return: float,
        returns_history: List[float],
        risk_aversion: float = 1.0
    ) -> float:
        """
        Calculate risk-adjusted returns as reward.

        Args:
            portfolio_return: Current portfolio return
            returns_history: List of historical returns
            risk_aversion: Risk aversion parameter

        Returns:
            Risk-adjusted return as reward
        """
        if len(returns_history) < 2:
            return portfolio_return
            
        # Calculate standard deviation of returns
        volatility = np.std(returns_history) + 1e-10
        
        # Calculate risk-adjusted return
        risk_adjusted = portfolio_return - risk_aversion * volatility
        
        return risk_adjusted

    @staticmethod
    def exponential_utility(
        portfolio_return: float,
        returns_history: List[float],
        risk_aversion: float = 1.0
    ) -> float:
        """
        Calculate exponential utility as reward.

        Args:
            portfolio_return: Current portfolio return
            returns_history: List of historical returns (not used in this function)
            risk_aversion: Risk aversion parameter

        Returns:
            Exponential utility as reward
        """
        # Exponential utility: U(r) = -exp(-λr)
        # Where λ is the risk aversion parameter
        # This function penalizes large negative returns more heavily
        
        # Scale return to avoid extreme values
        scaled_return = portfolio_return * 100  # Convert to percentage
        
        # Calculate exponential utility
        utility = -np.exp(-risk_aversion * scaled_return)
        
        # Normalize to a more reasonable range
        normalized_utility = (utility + 1) / 2  # Maps from [-1, 0] to [0, 0.5]
        
        return normalized_utility

    @staticmethod
    def get_reward_function(
        reward_type: str,
        risk_aversion: float = 1.0,
        risk_free_rate: float = 0.0
    ) -> Callable:
        """
        Get a reward function based on the specified type.

        Args:
            reward_type: Type of reward function
            risk_aversion: Risk aversion parameter
            risk_free_rate: Risk-free rate

        Returns:
            Reward function
        """
        if reward_type.lower() == 'returns':
            return lambda portfolio_value, previous_value, returns_history, portfolio_values: \
                RiskAdjustedReturns.simple_returns(portfolio_value, previous_value)
                
        elif reward_type.lower() == 'sharpe':
            return lambda portfolio_value, previous_value, returns_history, portfolio_values: \
                RiskAdjustedReturns.sharpe_ratio(returns_history, risk_free_rate)
                
        elif reward_type.lower() == 'sortino':
            return lambda portfolio_value, previous_value, returns_history, portfolio_values: \
                RiskAdjustedReturns.sortino_ratio(returns_history, risk_free_rate)
                
        elif reward_type.lower() == 'calmar':
            return lambda portfolio_value, previous_value, returns_history, portfolio_values: \
                RiskAdjustedReturns.calmar_ratio(returns_history, portfolio_values)
                
        elif reward_type.lower() == 'omega':
            return lambda portfolio_value, previous_value, returns_history, portfolio_values: \
                RiskAdjustedReturns.omega_ratio(returns_history)
                
        elif reward_type.lower() == 'risk_adjusted':
            return lambda portfolio_value, previous_value, returns_history, portfolio_values: \
                RiskAdjustedReturns.risk_adjusted_returns(
                    RiskAdjustedReturns.simple_returns(portfolio_value, previous_value),
                    returns_history,
                    risk_aversion
                )
                
        elif reward_type.lower() == 'exponential':
            return lambda portfolio_value, previous_value, returns_history, portfolio_values: \
                RiskAdjustedReturns.exponential_utility(
                    RiskAdjustedReturns.simple_returns(portfolio_value, previous_value),
                    returns_history,
                    risk_aversion
                )
                
        else:
            logger.warning(f"Unknown reward type: {reward_type}. Using simple returns.")
            return lambda portfolio_value, previous_value, returns_history, portfolio_values: \
                RiskAdjustedReturns.simple_returns(portfolio_value, previous_value) 