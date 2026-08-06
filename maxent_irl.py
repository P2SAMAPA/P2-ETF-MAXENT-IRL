"""
maxent_irl.py  —  Maximum Entropy Inverse Reinforcement Learning
=================================================================

Implements:
- Feature extraction from market data
- Maximum Entropy IRL for reward inference
- Trajectory sampling and optimization
- Reward function learning from observed behavior
"""

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


class FeatureExtractor:
    """
    Extract features from market data for IRL.
    
    Features represent different aspects of market behavior
    that the reward function might be optimizing for.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.use_momentum = config.get("momentum", True)
        self.use_volatility = config.get("volatility", True)
        self.use_skewness = config.get("skewness", True)
        self.use_drawdown = config.get("drawdown", True)
        self.use_risk_adjusted = config.get("risk_adjusted", True)
        self.use_macro_correlation = config.get("macro_correlation", True)
        
    def extract_features(self, returns: np.ndarray, macro: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from returns and macro data.
        
        Returns:
            feature_vector: (n_features,) array
        """
        features = []
        
        # Momentum features (different windows)
        if self.use_momentum:
            for w in [5, 10, 20, 60]:
                if len(returns) >= w:
                    features.append(np.mean(returns[-w:]))
                else:
                    features.append(0)
        
        # Volatility features
        if self.use_volatility:
            for w in [10, 20, 60]:
                if len(returns) >= w:
                    features.append(np.std(returns[-w:]))
                else:
                    features.append(0)
        
        # Skewness
        if self.use_skewness:
            if len(returns) >= 20:
                features.append(pd.Series(returns[-60:]).skew() if len(returns) >= 60 else 0)
            else:
                features.append(0)
        
        # Maximum drawdown
        if self.use_drawdown:
            if len(returns) >= 20:
                cum_returns = np.cumsum(returns[-60:]) if len(returns) >= 60 else np.cumsum(returns)
                running_max = np.maximum.accumulate(cum_returns)
                drawdown = running_max - cum_returns
                features.append(np.max(drawdown) if len(drawdown) > 0 else 0)
            else:
                features.append(0)
        
        # Risk-adjusted returns (Sharpe-like)
        if self.use_risk_adjusted:
            if len(returns) >= 20:
                mean_r = np.mean(returns[-60:]) if len(returns) >= 60 else np.mean(returns)
                std_r = np.std(returns[-60:]) if len(returns) >= 60 else np.std(returns)
                features.append(mean_r / (std_r + 1e-6))
            else:
                features.append(0)
        
        # Macro correlation
        if self.use_macro_correlation and len(macro) > 0:
            if len(returns) >= 20 and len(macro) >= 20:
                macro_flat = macro.flatten()[:5]
                for i in range(min(5, len(macro_flat))):
                    if np.std(returns) > 0 and np.std(macro_flat[i]) > 0:
                        corr = np.corrcoef(returns[-min(60, len(returns)):], 
                                         macro_flat[i][-min(60, len(macro_flat[i])):])[0, 1]
                        features.append(corr if not np.isnan(corr) else 0)
                    else:
                        features.append(0)
            else:
                features.extend([0] * 5)
        
        return np.array(features)


class MaximumEntropyIRL:
    """
    Maximum Entropy Inverse Reinforcement Learning.
    
    Learns the reward function that best explains observed behavior.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.n_iterations = config.get("n_iterations", 100)
        self.learning_rate = config.get("learning_rate", 0.01)
        self.convergence_threshold = config.get("convergence_threshold", 1e-4)
        self.regularization = config.get("regularization", 0.01)
        self.n_trajectories = config.get("n_trajectories", 50)
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(config)
        
        # Reward weights (learned)
        self.weights = None
        self.feature_names = []
        
    def compute_features(self, returns: np.ndarray, macro: np.ndarray) -> np.ndarray:
        """Compute features for a single time series."""
        return self.feature_extractor.extract_features(returns, macro)
    
    def compute_reward(self, features: np.ndarray) -> float:
        """Compute reward from features using learned weights."""
        if self.weights is None:
            return 0.0
        return np.dot(features, self.weights)
    
    def compute_trajectory_reward(self, returns: np.ndarray, macro: np.ndarray) -> float:
        """Compute total reward for a trajectory."""
        features = self.compute_features(returns, macro)
        return self.compute_reward(features)
    
    def sample_trajectories(self, returns: np.ndarray, macro: np.ndarray, 
                           n_trajectories: int = None) -> List[Dict]:
        """
        Sample trajectories from the data.
        
        A trajectory is a sequence of returns and macro observations.
        """
        if n_trajectories is None:
            n_trajectories = self.n_trajectories
        
        trajectories = []
        n_samples = len(returns)
        
        if n_samples < 20:
            return trajectories
        
        # Use a sliding window to create trajectories
        window_size = min(60, n_samples // 2)
        step_size = max(1, (n_samples - window_size) // n_trajectories)
        
        for start in range(0, n_samples - window_size, step_size):
            end = start + window_size
            if end <= n_samples:
                traj_returns = returns[start:end]
                traj_macro = macro[start:end] if len(macro) > 0 else np.zeros((window_size, 1))
                
                # Compute features for this trajectory
                features = self.compute_features(traj_returns, traj_macro)
                
                trajectories.append({
                    "returns": traj_returns,
                    "macro": traj_macro,
                    "features": features,
                    "start": start,
                    "end": end
                })
                
                if len(trajectories) >= n_trajectories:
                    break
        
        return trajectories
    
    def compute_expected_features(self, trajectories: List[Dict]) -> np.ndarray:
        """Compute expected feature counts across trajectories."""
        if not trajectories:
            return np.zeros(1)
        
        n_features = len(trajectories[0]["features"])
        expected_features = np.zeros(n_features)
        
        for traj in trajectories:
            expected_features += traj["features"]
        
        return expected_features / len(trajectories)
    
    def compute_partition_function(self, trajectories: List[Dict], weights: np.ndarray) -> float:
        """Compute the partition function Z = Σ exp(θ · f)."""
        if not trajectories:
            return 1.0
        
        rewards = []
        for traj in trajectories:
            reward = np.dot(traj["features"], weights)
            rewards.append(reward)
        
        # Subtract max for numerical stability
        rewards = np.array(rewards)
        max_reward = np.max(rewards)
        exp_rewards = np.exp(rewards - max_reward)
        
        return np.sum(exp_rewards) + 1e-8
    
    def compute_gradient(self, trajectories: List[Dict], weights: np.ndarray) -> np.ndarray:
        """Compute gradient of the log-likelihood."""
        if not trajectories:
            return np.zeros(1)
        
        n_features = len(trajectories[0]["features"])
        gradient = np.zeros(n_features)
        
        # Expected features under current weights
        expected_features = np.zeros(n_features)
        partition = self.compute_partition_function(trajectories, weights)
        
        for traj in trajectories:
            exp_reward = np.exp(np.dot(traj["features"], weights))
            prob = exp_reward / partition
            expected_features += prob * traj["features"]
        
        # Empirical feature counts
        empirical_features = self.compute_expected_features(trajectories)
        
        # Gradient = empirical - expected
        gradient = empirical_features - expected_features
        
        # Add L2 regularization
        gradient -= self.regularization * weights
        
        return gradient
    
    def train(self, returns: np.ndarray, macro: np.ndarray) -> Dict:
        """
        Train the IRL model to learn reward weights.
        
        Returns:
            weights: Learned reward weights
            history: Training history
        """
        # Sample trajectories
        trajectories = self.sample_trajectories(returns, macro)
        
        if not trajectories:
            return {
                "weights": np.zeros(1),
                "history": [],
                "converged": False,
                "error": "No trajectories sampled"
            }
        
        n_features = len(trajectories[0]["features"])
        self.weights = np.random.randn(n_features) * 0.01
        
        history = []
        
        for iteration in range(self.n_iterations):
            # Compute gradient
            gradient = self.compute_gradient(trajectories, self.weights)
            
            # Update weights
            self.weights += self.learning_rate * gradient
            
            # Compute loss
            loss = -np.sum([np.dot(traj["features"], self.weights) for traj in trajectories])
            loss += self.regularization * np.sum(self.weights ** 2)
            loss += np.log(self.compute_partition_function(trajectories, self.weights))
            
            history.append({
                "iteration": iteration,
                "loss": loss,
                "weights": self.weights.copy(),
                "gradient_norm": np.linalg.norm(gradient)
            })
            
            # Check convergence
            if len(history) > 10:
                recent_losses = [h["loss"] for h in history[-10:]]
                if max(recent_losses) - min(recent_losses) < self.convergence_threshold:
                    break
        
        return {
            "weights": self.weights,
            "history": history,
            "converged": len(history) < self.n_iterations,
            "n_iterations": len(history),
            "n_trajectories": len(trajectories)
        }
    
    def predict_reward(self, returns: np.ndarray, macro: np.ndarray) -> float:
        """Predict reward for a new sequence."""
        features = self.compute_features(returns, macro)
        return self.compute_reward(features)


def compute_maxent_irl(
    prices: pd.Series,
    macro_df: pd.DataFrame,
    config: Dict,
    window: int = 252
) -> Dict:
    """
    Compute Maximum Entropy IRL signals for a single ticker.
    """
    returns = np.log(prices / prices.shift(1)).dropna().values
    macro = macro_df.values
    
    if len(returns) < window:
        return {
            "reward": 0,
            "z_score": 0,
            "weights": [],
            "error": "Insufficient data"
        }
    
    try:
        # Use recent window
        train_returns = returns[-window:]
        train_macro = macro[-min(window, len(macro)):] if len(macro) > 0 else np.zeros((1, 6))
        
        # Initialize IRL
        irl = MaximumEntropyIRL(config)
        
        # Train
        result = irl.train(train_returns, train_macro)
        
        if result.get("error"):
            return {
                "reward": 0,
                "z_score": 0,
                "weights": [],
                "error": result["error"]
            }
        
        # Compute reward
        reward = irl.predict_reward(train_returns, train_macro)
        
        return {
            "reward": reward,
            "z_score": reward,  # Will be normalized
            "weights": irl.weights.tolist() if irl.weights is not None else [],
            "n_iterations": result.get("n_iterations", 0),
            "converged": result.get("converged", False),
            "n_trajectories": result.get("n_trajectories", 0),
            "error": None
        }
    except Exception as e:
        return {
            "reward": 0,
            "z_score": 0,
            "weights": [],
            "error": str(e)
        }


def compute_universe_maxent_irl(
    prices_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    config: Dict,
    window: int = 252
) -> Dict:
    """
    Compute Maximum Entropy IRL for all ETFs in a universe.
    """
    results = {}
    
    for ticker in prices_df.columns:
        prices = prices_df[ticker]
        result = compute_maxent_irl(prices, macro_df, config, window)
        
        results[ticker] = {
            "reward": result.get("reward", 0),
            "z_score": result.get("z_score", 0),
            "weights": result.get("weights", []),
            "n_iterations": result.get("n_iterations", 0),
            "converged": result.get("converged", False),
            "n_trajectories": result.get("n_trajectories", 0)
        }
    
    # Normalize z-scores
    z_values = np.array([r["z_score"] for r in results.values()])
    if len(z_values) > 1 and np.std(z_values) > 1e-6:
        mean_z = np.mean(z_values)
        std_z = np.std(z_values)
        for ticker, r in results.items():
            r["z_score"] = (r["z_score"] - mean_z) / std_z
    else:
        # Use reward as fallback
        rewards = np.array([r["reward"] for r in results.values()])
        if len(rewards) > 1 and np.std(rewards) > 1e-6:
            mean_r = np.mean(rewards)
            std_r = np.std(rewards)
            for ticker, r in results.items():
                r["z_score"] = (r["reward"] - mean_r) / std_r
        else:
            for r in results.values():
                r["z_score"] = 0
    
    return results
