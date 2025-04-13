"""
Portfolio Agent Module.

This module implements a reinforcement learning agent for portfolio optimization
using various RL algorithms.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import pickle
from datetime import datetime
import time
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

from .trading_environment import TradingEnvironment

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioAgent:
    """
    Reinforcement learning agent for portfolio optimization.
    
    This class implements various RL algorithms for portfolio optimization,
    including training, evaluation, and inference.
    """

    def __init__(
        self,
        algorithm: str = 'ppo',
        policy: str = 'MlpPolicy',
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        seed: int = 42,
        tensorboard_log: Optional[str] = None,
        device: str = 'auto'
    ):
        """
        Initialize the portfolio agent.

        Args:
            algorithm: RL algorithm to use ('ppo', 'a2c', or 'sac')
            policy: Policy network architecture
            learning_rate: Learning rate
            gamma: Discount factor
            seed: Random seed
            tensorboard_log: Directory for TensorBoard logs
            device: Device to run on ('auto', 'cpu', or 'cuda')
        """
        self.algorithm = algorithm.lower()
        self.policy = policy
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.seed = seed
        self.tensorboard_log = tensorboard_log
        self.device = device
        
        self.model = None
        self.env = None
        self.vec_env = None
        
        logger.info(f"PortfolioAgent initialized with {algorithm} algorithm")

    def create_env(
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
        risk_aversion: float = 1.0,
        monitor_log_dir: Optional[str] = None,
        normalize_env: bool = True
    ) -> Union[TradingEnvironment, DummyVecEnv]:
        """
        Create and configure the trading environment.

        Args:
            df: DataFrame with market data
            tickers: List of stock symbols
            initial_balance: Initial cash balance
            transaction_cost_pct: Transaction cost as percentage
            window_size: Number of time steps to include in state
            reward_function: Type of reward function
            max_steps: Maximum number of steps per episode
            tech_indicators: Whether to include technical indicators
            sentiment_features: Whether to include sentiment features
            risk_aversion: Risk aversion parameter
            monitor_log_dir: Directory for environment monitoring logs
            normalize_env: Whether to normalize the environment

        Returns:
            Configured environment
        """
        # Create trading environment
        env = TradingEnvironment(
            df=df,
            tickers=tickers,
            initial_balance=initial_balance,
            transaction_cost_pct=transaction_cost_pct,
            window_size=window_size,
            reward_function=reward_function,
            max_steps=max_steps,
            tech_indicators=tech_indicators,
            sentiment_features=sentiment_features,
            risk_aversion=risk_aversion
        )
        
        self.env = env
        
        # Wrap environment with Monitor if log directory provided
        if monitor_log_dir:
            os.makedirs(monitor_log_dir, exist_ok=True)
            env = Monitor(env, os.path.join(monitor_log_dir, f"{int(time.time())}"))
            
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: env])
        
        # Normalize observations and rewards if requested
        if normalize_env:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0
            )
            
        self.vec_env = vec_env
        return vec_env

    def build_model(self) -> Any:
        """
        Build the RL model based on the selected algorithm.

        Returns:
            Stable Baselines model
        """
        if self.vec_env is None:
            raise ValueError("Environment must be created before building the model")
            
        # Create model based on algorithm
        if self.algorithm == 'ppo':
            model = PPO(
                policy=self.policy,
                env=self.vec_env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                seed=self.seed,
                tensorboard_log=self.tensorboard_log,
                device=self.device,
                verbose=1
            )
        elif self.algorithm == 'a2c':
            model = A2C(
                policy=self.policy,
                env=self.vec_env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                seed=self.seed,
                tensorboard_log=self.tensorboard_log,
                device=self.device,
                verbose=1
            )
        elif self.algorithm == 'sac':
            model = SAC(
                policy=self.policy,
                env=self.vec_env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                seed=self.seed,
                tensorboard_log=self.tensorboard_log,
                device=self.device,
                verbose=1
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
        self.model = model
        logger.info(f"Created {self.algorithm.upper()} model")
        return model

    def train(
        self,
        total_timesteps: int = 10000,
        eval_freq: int = 1000,
        n_eval_episodes: int = 5,
        eval_env: Optional[Any] = None,
        callback: Optional[BaseCallback] = None,
        save_dir: Optional[str] = None,
        save_freq: int = 5000
    ) -> Any:
        """
        Train the agent.

        Args:
            total_timesteps: Total number of timesteps for training
            eval_freq: Frequency of evaluation during training
            n_eval_episodes: Number of episodes for evaluation
            eval_env: Optional separate environment for evaluation
            callback: Optional callback for training
            save_dir: Directory to save model checkpoints
            save_freq: Frequency of saving checkpoints

        Returns:
            Trained model
        """
        if self.model is None:
            self.build_model()
            
        # Set up evaluation environment
        if eval_env is None and self.env is not None:
            eval_env = DummyVecEnv([lambda: self.env])
            
        # Set up evaluation callback if save_dir is provided
        callbacks = []
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Create eval callback
            if eval_env is not None:
                eval_callback = EvalCallback(
                    eval_env=eval_env,
                    best_model_save_path=save_dir,
                    log_path=save_dir,
                    eval_freq=eval_freq,
                    deterministic=True,
                    render=False,
                    n_eval_episodes=n_eval_episodes
                )
                callbacks.append(eval_callback)
                
            # Create checkpoint callback
            class CheckpointCallback(BaseCallback):
                def __init__(self, save_dir, save_freq):
                    super(CheckpointCallback, self).__init__()
                    self.save_dir = save_dir
                    self.save_freq = save_freq
                    
                def _on_step(self):
                    if self.n_calls % self.save_freq == 0:
                        model_path = os.path.join(
                            self.save_dir, 
                            f"model_{self.n_calls}_steps.zip"
                        )
                        self.model.save(model_path)
                        if hasattr(self.training_env, 'save'):
                            env_path = os.path.join(
                                self.save_dir, 
                                f"env_{self.n_calls}_steps.pkl"
                            )
                            self.training_env.save(env_path)
                    return True
                    
            checkpoint_callback = CheckpointCallback(save_dir, save_freq)
            callbacks.append(checkpoint_callback)
            
        # Add user-provided callback
        if callback is not None:
            callbacks.append(callback)
            
        # Create callback list
        if callbacks:
            from stable_baselines3.common.callbacks import CallbackList
            callback = CallbackList(callbacks)
        else:
            callback = None
            
        # Train the model
        logger.info(f"Starting training for {total_timesteps} timesteps")
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        
        # Save final model
        if save_dir:
            final_model_path = os.path.join(save_dir, "final_model.zip")
            self.model.save(final_model_path)
            logger.info(f"Final model saved to {final_model_path}")
            
            if hasattr(self.vec_env, 'save'):
                final_env_path = os.path.join(save_dir, "final_env.pkl")
                self.vec_env.save(final_env_path)
                logger.info(f"Final environment saved to {final_env_path}")
                
        return self.model

    def evaluate(
        self,
        env: Optional[Any] = None,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False
    ) -> Tuple[float, float]:
        """
        Evaluate the trained agent.

        Args:
            env: Environment for evaluation
            n_eval_episodes: Number of episodes for evaluation
            deterministic: Whether to use deterministic actions
            render: Whether to render the environment during evaluation

        Returns:
            Tuple of (mean_reward, std_reward)
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        # Use provided environment or default
        eval_env = env if env is not None else self.vec_env
        
        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            render=render
        )
        
        logger.info(f"Evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Any]:
        """
        Make a prediction with the trained agent.

        Args:
            observation: Observation from the environment
            deterministic: Whether to use deterministic actions

        Returns:
            Tuple of (action, state)
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
            
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str) -> None:
        """
        Save the agent, including model and environment.

        Args:
            path: Path to save the agent
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        model_path = f"{path}_model.zip"
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save environment if it has a save method
        if hasattr(self.vec_env, 'save'):
            env_path = f"{path}_env.pkl"
            self.vec_env.save(env_path)
            logger.info(f"Environment saved to {env_path}")

    def load(
        self,
        model_path: str,
        env_path: Optional[str] = None
    ) -> None:
        """
        Load a saved agent.

        Args:
            model_path: Path to the saved model
            env_path: Path to the saved environment
        """
        # Load model
        if self.algorithm == 'ppo':
            self.model = PPO.load(model_path)
        elif self.algorithm == 'a2c':
            self.model = A2C.load(model_path)
        elif self.algorithm == 'sac':
            self.model = SAC.load(model_path)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
        logger.info(f"Model loaded from {model_path}")
        
        # Load environment if provided
        if env_path and os.path.exists(env_path):
            self.vec_env = VecNormalize.load(env_path, self.vec_env)
            logger.info(f"Environment loaded from {env_path}")

    def backtest(
        self,
        df: pd.DataFrame,
        tickers: List[str],
        initial_balance: float = 10000.0,
        transaction_cost_pct: float = 0.001,
        window_size: int = 10
    ) -> Dict:
        """
        Backtest the trained agent on historical data.

        Args:
            df: DataFrame with historical data
            tickers: List of stock symbols
            initial_balance: Initial cash balance
            transaction_cost_pct: Transaction cost as percentage
            window_size: Number of time steps to include in state

        Returns:
            Dictionary with backtest results
        """
        if self.model is None:
            raise ValueError("Model must be trained before backtesting")
            
        # Create environment for backtesting
        env = TradingEnvironment(
            df=df,
            tickers=tickers,
            initial_balance=initial_balance,
            transaction_cost_pct=transaction_cost_pct,
            window_size=window_size,
            reward_function='returns'  # Use returns for backtesting
        )
        
        # Run backtest
        observation = env.reset()
        done = False
        portfolio_values = [initial_balance]
        actions = []
        rewards = []
        
        while not done:
            action, _state = self.model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)
            
            portfolio_values.append(info['portfolio_value'])
            actions.append(action)
            rewards.append(reward)
            
        # Calculate performance metrics
        returns = np.array(portfolio_values[1:]) / np.array(portfolio_values[:-1]) - 1
        cumulative_return = portfolio_values[-1] / portfolio_values[0] - 1
        annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
        annualized_volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.max(portfolio_values)
        
        # Prepare results
        results = {
            'portfolio_values': portfolio_values,
            'returns': returns.tolist(),
            'actions': [a.tolist() for a in actions],
            'rewards': rewards,
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'initial_balance': initial_balance,
            'final_balance': portfolio_values[-1],
            'tickers': tickers,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Backtest completed: return={cumulative_return:.2%}, Sharpe={sharpe_ratio:.2f}")
        return results

    def get_optimal_portfolio(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Dict:
        """
        Get the optimal portfolio allocation for a given observation.

        Args:
            observation: Observation from the environment
            deterministic: Whether to use deterministic actions

        Returns:
            Dictionary with portfolio allocation
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting optimal portfolio")
            
        if self.env is None:
            raise ValueError("Environment must be created before getting optimal portfolio")
            
        # Get action from model
        action, _state = self.model.predict(observation, deterministic=deterministic)
        
        # Normalize action to ensure it sums to 1
        action = action / (np.sum(action) + 1e-10)
        
        # Create portfolio allocation dictionary
        portfolio = {
            'weights': action.tolist(),
            'assets': self.env.tickers + ['cash'],
            'timestamp': datetime.now().isoformat()
        }
        
        return portfolio 