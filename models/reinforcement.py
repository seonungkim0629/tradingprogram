"""
Reinforcement Learning Models for Bitcoin Trading Bot

This module implements PPO-based reinforcement learning models for trading.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from datetime import datetime
import json
import matplotlib.pyplot as plt

from models.base import ReinforcementLearningModel
from utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class TradingEnvironment(gym.Env):
    """Custom Gym environment for Bitcoin trading"""
    
    def __init__(self, 
                data: pd.DataFrame,
                window_size: int = 30,
                initial_balance: float = 10000.0,
                commission: float = 0.001,
                reward_scaling: float = 0.01,
                features: Optional[List[str]] = None,
                render_mode: Optional[str] = None):
        """
        Initialize the trading environment
        
        Args:
            data (pd.DataFrame): Historical price data with OHLCV and indicators
            window_size (int): Number of past candles to use as observation
            initial_balance (float): Initial portfolio balance in USD
            commission (float): Trading commission as a fraction (0.001 = 0.1%)
            reward_scaling (float): Scaling factor for rewards
            features (Optional[List[str]]): List of features to use in observation space
            render_mode (Optional[str]): The render mode to use ('human', 'rgb_array', None)
        """
        super(TradingEnvironment, self).__init__()
        
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_scaling = reward_scaling
        self.render_mode = render_mode
        
        # Define features to use
        if features is None:
            # Default features: use all columns except date
            self.features = [col for col in data.columns if col != 'date' and col != 'timestamp']
        else:
            self.features = features
        
        self.num_features = len(self.features)
        
        # Define action space: 0 = Sell, 1 = Hold, 2 = Buy
        self.action_space = spaces.Discrete(3)
        
        # Define observation space: [window_size, num_features + positions + balance]
        # Features + current BTC position + current USD balance
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size, self.num_features + 2),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
        logger.info(f"Trading environment initialized with {self.num_features} features and window size {window_size}")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state
        
        Args:
            seed (Optional[int]): Random seed
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Initial observation and info
        """
        super().reset(seed=seed)
        
        # Set initial position in the time series
        self.current_step = self.window_size
        
        # Set initial portfolio
        self.balance = self.initial_balance
        self.btc_held = 0.0
        self.current_price = self.data.iloc[self.current_step]['close']
        
        # Calculate initial portfolio value
        self.portfolio_value = self.balance + self.btc_held * self.current_price
        
        # Trading history
        self.trades = []
        self.portfolio_values = [self.portfolio_value]
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment based on the action
        
        Args:
            action (int): 0 = Sell, 1 = Hold, 2 = Buy
            
        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: Observation, reward, terminated, truncated, info
        """
        # Get current price
        self.current_price = self.data.iloc[self.current_step]['close']
        
        # Calculate portfolio value before action
        portfolio_value_before = self.balance + self.btc_held * self.current_price
        
        # Execute action
        reward = 0
        
        if action == 0:  # Sell
            if self.btc_held > 0:
                # Calculate sell amount (all)
                sell_amount = self.btc_held * self.current_price * (1 - self.commission)
                
                # Update holdings
                self.balance += sell_amount
                self.btc_held = 0
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'price': self.current_price,
                    'type': 'sell',
                    'amount': sell_amount,
                    'commission': sell_amount * self.commission / (1 - self.commission)
                })
                
                logger.debug(f"Step {self.current_step}: Sold all BTC at {self.current_price}")
        
        elif action == 2:  # Buy
            if self.balance > 0:
                # Calculate buy amount (all)
                buy_amount_btc = (self.balance / self.current_price) * (1 - self.commission)
                
                # Update holdings
                self.btc_held += buy_amount_btc
                self.balance = 0
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'price': self.current_price,
                    'type': 'buy',
                    'amount': buy_amount_btc,
                    'commission': buy_amount_btc * self.current_price * self.commission / (1 - self.commission)
                })
                
                logger.debug(f"Step {self.current_step}: Bought {buy_amount_btc:.8f} BTC at {self.current_price}")
        
        # Move to next step
        self.current_step += 1
        
        # Calculate portfolio value after action
        self.current_price = self.data.iloc[self.current_step]['close']
        portfolio_value_after = self.balance + self.btc_held * self.current_price
        
        # Update portfolio history
        self.portfolio_value = portfolio_value_after
        self.portfolio_values.append(portfolio_value_after)
        
        # Calculate reward: change in portfolio value
        reward = (portfolio_value_after - portfolio_value_before) * self.reward_scaling
        
        # Determine if episode is terminated
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation
        
        Returns:
            np.ndarray: Observation matrix
        """
        # Get window of feature data
        start_idx = self.current_step - self.window_size + 1
        end_idx = self.current_step + 1
        
        # Extract features
        features_window = self.data.iloc[start_idx:end_idx][self.features].values
        
        # Normalize features
        price = self.data.iloc[self.current_step]['close']
        features_normalized = features_window.copy()
        
        # Add position and balance info
        positions = np.zeros((self.window_size, 2))
        
        # Fill the last row with current position
        positions[-1, 0] = self.btc_held / (self.initial_balance / price)  # Normalized BTC held
        positions[-1, 1] = self.balance / self.initial_balance  # Normalized balance
        
        # Combine features with positions
        observation = np.hstack([features_normalized, positions])
        
        return observation.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get information about current state
        
        Returns:
            Dict[str, Any]: Info dictionary
        """
        return {
            'step': self.current_step,
            'price': self.current_price,
            'balance': self.balance,
            'btc_held': self.btc_held,
            'portfolio_value': self.portfolio_value,
            'return': (self.portfolio_value / self.initial_balance) - 1.0,
            'num_trades': len(self.trades)
        }
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment
        
        Returns:
            Optional[np.ndarray]: Rendered frame if render_mode is 'rgb_array'
        """
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            # Create a plot showing portfolio value
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Plot price
            steps = range(self.window_size, self.current_step + 1)
            ax1.plot(steps, self.data.iloc[self.window_size:self.current_step + 1]['close'], label='BTC Price')
            ax1.set_title('Bitcoin Price')
            ax1.legend()
            
            # Plot portfolio value
            ax2.plot(steps, self.portfolio_values, label='Portfolio Value')
            ax2.set_title('Portfolio Value')
            ax2.legend()
            
            # Mark trades
            for trade in self.trades:
                marker = 'v' if trade['type'] == 'sell' else '^'
                color = 'r' if trade['type'] == 'sell' else 'g'
                ax1.scatter(trade['step'], trade['price'], color=color, marker=marker, s=100)
            
            plt.tight_layout()
            
            if self.render_mode == "human":
                plt.pause(0.01)
                plt.show()
                return None
            elif self.render_mode == "rgb_array":
                fig.canvas.draw()
                img = np.array(fig.canvas.renderer.buffer_rgba())
                plt.close()
                return img
        
        return None


class PPOTrader(ReinforcementLearningModel):
    """PPO-based Reinforcement Learning model for Bitcoin trading"""
    
    def __init__(self, 
                name: str = "PPOTrader", 
                version: str = "1.0.0",
                action_space_size: int = 3,
                window_size: int = 30,
                feature_dim: int = 10,
                actor_units: List[int] = [64, 32],
                critic_units: List[int] = [64, 32],
                learning_rate: float = 0.0003,
                gamma: float = 0.99,
                clip_ratio: float = 0.2,
                epochs: int = 10,
                batch_size: int = 64):
        """
        Initialize PPO Trader model
        
        Args:
            name (str): Model name
            version (str): Model version
            action_space_size (int): Size of action space (3 for Sell/Hold/Buy)
            window_size (int): Number of past candles to use as observation
            feature_dim (int): Number of features in each observation step
            actor_units (List[int]): List of units in actor network
            critic_units (List[int]): List of units in critic network
            learning_rate (float): Learning rate for Adam optimizer
            gamma (float): Discount factor for future rewards
            clip_ratio (float): Clip ratio for PPO
            epochs (int): Number of epochs for each PPO update
            batch_size (int): Batch size for training
        """
        super().__init__(name, version)
        
        self.action_space_size = action_space_size
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.actor_units = actor_units
        self.critic_units = critic_units
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Store parameters
        self.params = {
            'action_space_size': action_space_size,
            'window_size': window_size,
            'feature_dim': feature_dim,
            'actor_units': actor_units,
            'critic_units': critic_units,
            'learning_rate': learning_rate,
            'gamma': gamma,
            'clip_ratio': clip_ratio,
            'epochs': epochs,
            'batch_size': batch_size
        }
        
        # Build models
        self.build_models()
        
        # Training buffers
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        self.buffer_size = 0
        
        self.logger.info(f"Initialized {self.name} model")
    
    def build_models(self) -> None:
        """Build actor and critic networks"""
        # Actor network (policy network)
        self.actor = self._build_actor()
        
        # Critic network (value network)
        self.critic = self._build_critic()
        
        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
    
    def _build_actor(self) -> keras.Model:
        """
        Build actor network
        
        Returns:
            keras.Model: Actor model
        """
        inputs = layers.Input(shape=(self.window_size, self.feature_dim))
        
        # Flatten the time series
        x = layers.Flatten()(inputs)
        
        # Hidden layers
        for units in self.actor_units:
            x = layers.Dense(units, activation='relu')(x)
        
        # Output layer (categorical action distribution)
        logits = layers.Dense(self.action_space_size, activation=None)(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=logits, name='actor')
        return model
    
    def _build_critic(self) -> keras.Model:
        """
        Build critic network
        
        Returns:
            keras.Model: Critic model
        """
        inputs = layers.Input(shape=(self.window_size, self.feature_dim))
        
        # Flatten the time series
        x = layers.Flatten()(inputs)
        
        # Hidden layers
        for units in self.critic_units:
            x = layers.Dense(units, activation='relu')(x)
        
        # Output layer (state value)
        value = layers.Dense(1, activation=None)(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=value, name='critic')
        return model
    
    def _log_prob(self, logits: tf.Tensor, action: int) -> tf.Tensor:
        """
        Calculate log probability of action given logits
        
        Args:
            logits (tf.Tensor): Action logits
            action (int): Action taken
            
        Returns:
            tf.Tensor: Log probability
        """
        action_mask = tf.one_hot(action, self.action_space_size)
        log_probs = tf.nn.log_softmax(logits)
        return tf.reduce_sum(log_probs * action_mask, axis=1)
    
    def act(self, 
           state: np.ndarray, 
           deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Choose an action based on the current state
        
        Args:
            state (np.ndarray): Current observation
            deterministic (bool): Whether to take deterministic action
            
        Returns:
            Tuple[int, float, float]: Action, log probability, value
        """
        # Reshape state for model input
        state_input = np.expand_dims(state, axis=0)
        
        # Get action logits and value
        logits = self.actor(state_input, training=False)[0]
        value = self.critic(state_input, training=False)[0, 0]
        
        if deterministic:
            # Take action with highest probability
            action = tf.argmax(logits).numpy()
            action_probs = tf.nn.softmax(logits).numpy()
            log_prob = np.log(action_probs[action])
        else:
            # Sample action from probability distribution
            action_probs = tf.nn.softmax(logits).numpy()
            action = np.random.choice(self.action_space_size, p=action_probs)
            log_prob = np.log(action_probs[action])
        
        return action, log_prob, value
    
    def update(self, 
              states: np.ndarray, 
              actions: np.ndarray, 
              old_log_probs: np.ndarray,
              rewards: np.ndarray, 
              values: np.ndarray, 
              dones: np.ndarray) -> Dict[str, float]:
        """
        Update model using PPO algorithm
        
        Args:
            states (np.ndarray): State observations
            actions (np.ndarray): Actions taken
            old_log_probs (np.ndarray): Log probabilities of actions when they were taken
            rewards (np.ndarray): Rewards received
            values (np.ndarray): Value estimates
            dones (np.ndarray): Done flags
            
        Returns:
            Dict[str, float]: Training metrics
        """
        # Calculate advantages
        returns, advantages = self._compute_advantages(rewards, values, dones)
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        
        # Normalize advantages
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        
        # Training metrics
        metrics = {
            'actor_loss': 0,
            'critic_loss': 0,
            'entropy': 0,
            'kl_divergence': 0
        }
        
        # Mini-batch training
        dataset = tf.data.Dataset.from_tensor_slices((states, actions, old_log_probs, returns, advantages))
        dataset = dataset.shuffle(buffer_size=len(states)).batch(self.batch_size)
        
        for _ in range(self.epochs):
            for batch in dataset:
                batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages = batch
                
                # Update networks
                batch_metrics = self._update_batch(
                    batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages
                )
                
                # Update metrics
                for key in metrics:
                    metrics[key] += batch_metrics[key] / self.epochs
        
        return metrics 
    
    def _compute_advantages(self, 
                          rewards: np.ndarray, 
                          values: np.ndarray, 
                          dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages and returns using Generalized Advantage Estimation (GAE)
        
        Args:
            rewards (np.ndarray): Array of rewards
            values (np.ndarray): Array of value estimates
            dones (np.ndarray): Array of done flags
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Returns and advantages
        """
        length = len(rewards)
        returns = np.zeros(length)
        advantages = np.zeros(length)
        last_value = 0
        last_advantage = 0
        
        # Compute returns and advantages going backward
        for t in reversed(range(length)):
            # If it's the last step, next value is 0
            # Otherwise, it's the value of the next state
            next_value = last_value if t == length - 1 or dones[t] else values[t + 1]
            
            # Compute TD target
            target = rewards[t] + self.gamma * next_value * (1 - dones[t])
            
            # Compute TD error (advantage)
            delta = target - values[t]
            
            # Compute advantage using GAE
            last_advantage = delta + self.gamma * (1 - dones[t]) * last_advantage
            advantages[t] = last_advantage
            
            # Compute return
            returns[t] = target
            
            # Update last value
            last_value = values[t]
        
        return returns, advantages
    
    @tf.function
    def _update_batch(self, 
                     states: tf.Tensor, 
                     actions: tf.Tensor, 
                     old_log_probs: tf.Tensor, 
                     returns: tf.Tensor, 
                     advantages: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Update model for a single batch using PPO algorithm
        
        Args:
            states (tf.Tensor): State observations
            actions (tf.Tensor): Actions taken
            old_log_probs (tf.Tensor): Log probabilities of actions when they were taken
            returns (tf.Tensor): Discounted returns
            advantages (tf.Tensor): Advantages
            
        Returns:
            Dict[str, tf.Tensor]: Training metrics
        """
        with tf.GradientTape() as tape:
            # Forward pass through actor and critic
            logits = self.actor(states)
            values = self.critic(states)[:, 0]
            
            # Calculate log probabilities of actions
            action_masks = tf.one_hot(actions, self.action_space_size)
            log_probs = tf.reduce_sum(tf.nn.log_softmax(logits) * action_masks, axis=1)
            
            # Calculate entropy (for exploration)
            entropy = -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(logits) * tf.nn.log_softmax(logits), axis=1))
            
            # Calculate policy ratio
            ratio = tf.exp(log_probs - old_log_probs)
            
            # Calculate KL divergence for monitoring
            kl_divergence = tf.reduce_mean(old_log_probs - log_probs)
            
            # Calculate surrogate loss
            surrogate1 = ratio * advantages
            surrogate2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            
            # Calculate value loss
            critic_loss = tf.reduce_mean(tf.square(returns - values))
            
            # Combine losses
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        # Calculate gradients
        gradients = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables + self.critic.trainable_variables))
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy': entropy,
            'kl_divergence': kl_divergence
        }
    
    def train(self, 
             env: TradingEnvironment, 
             num_episodes: int = 100,
             max_steps_per_episode: int = 1000,
             save_freq: int = 10,
             eval_freq: int = 10,
             **kwargs) -> Dict[str, Any]:
        """
        Train the model using the provided environment
        
        Args:
            env (TradingEnvironment): Trading environment
            num_episodes (int): Number of episodes to train
            max_steps_per_episode (int): Maximum steps per episode
            save_freq (int): Frequency of model saving (episodes)
            eval_freq (int): Frequency of evaluation (episodes)
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Training metrics
        """
        start_time = pd.Timestamp.now()
        self.logger.info(f"Starting training for {num_episodes} episodes")
        
        # Training metrics
        episode_rewards = []
        episode_lengths = []
        episode_returns = []
        update_metrics = []
        
        # Main training loop
        for episode in range(1, num_episodes + 1):
            # Reset environment
            state, info = env.reset()
            
            # Initialize episode variables
            episode_reward = 0
            episode_length = 0
            
            # Clear buffer
            self.buffer = {
                'states': [],
                'actions': [],
                'rewards': [],
                'values': [],
                'log_probs': [],
                'dones': []
            }
            self.buffer_size = 0
            
            # Run episode
            for step in range(max_steps_per_episode):
                # Choose action
                action, log_prob, value = self.act(state)
                
                # Take action in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store transition in buffer
                self.buffer['states'].append(state)
                self.buffer['actions'].append(action)
                self.buffer['rewards'].append(reward)
                self.buffer['values'].append(value)
                self.buffer['log_probs'].append(log_prob)
                self.buffer['dones'].append(done)
                self.buffer_size += 1
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # End episode if done
                if done:
                    break
            
            # Update model
            metrics = self.update(
                np.array(self.buffer['states']),
                np.array(self.buffer['actions']),
                np.array(self.buffer['log_probs']),
                np.array(self.buffer['rewards']),
                np.array(self.buffer['values']),
                np.array(self.buffer['dones'])
            )
            
            # Store episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_returns.append(info['return'])
            update_metrics.append(metrics)
            
            # Log progress
            self.logger.info(f"Episode {episode}/{num_episodes} - Reward: {episode_reward:.2f}, Return: {info['return']:.2f}, Length: {episode_length}")
            
            # Save model periodically
            if episode % save_freq == 0:
                self.save_model(f"{self.name}_episode_{episode}")
            
            # Evaluate model periodically
            if episode % eval_freq == 0:
                eval_metrics = self.evaluate(env, num_episodes=3)
                self.logger.info(f"Evaluation after episode {episode} - Avg Return: {eval_metrics['avg_return']:.2f}")
        
        # Calculate overall metrics
        training_metrics = {
            'num_episodes': num_episodes,
            'avg_reward': np.mean(episode_rewards),
            'avg_return': np.mean(episode_returns),
            'avg_length': np.mean(episode_lengths),
            'training_time': (pd.Timestamp.now() - start_time).total_seconds()
        }
        
        # Update model metrics
        self.metrics.update(training_metrics)
        self.is_trained = True
        self.last_update = pd.Timestamp.now()
        
        self.logger.info(f"Training completed. Avg Return: {training_metrics['avg_return']:.2f}")
        return training_metrics
    
    def evaluate(self, 
                env: TradingEnvironment, 
                num_episodes: int = 5,
                deterministic: bool = True,
                **kwargs) -> Dict[str, Any]:
        """
        Evaluate the model
        
        Args:
            env (TradingEnvironment): Trading environment
            num_episodes (int): Number of episodes to evaluate
            deterministic (bool): Whether to take deterministic actions
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        self.logger.info(f"Evaluating model for {num_episodes} episodes")
        
        # Evaluation metrics
        episode_rewards = []
        episode_lengths = []
        episode_returns = []
        episode_trades = []
        
        # Run evaluation episodes
        for episode in range(1, num_episodes + 1):
            # Reset environment
            state, info = env.reset()
            
            # Initialize episode variables
            episode_reward = 0
            episode_length = 0
            
            # Run episode
            done = False
            while not done:
                # Choose action
                action, _, _ = self.act(state, deterministic=deterministic)
                
                # Take action in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Update state and metrics
                state = next_state
                episode_reward += reward
                episode_length += 1
            
            # Store episode metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_returns.append(info['return'])
            episode_trades.append(info['num_trades'])
            
            self.logger.info(f"Eval Episode {episode}/{num_episodes} - Return: {info['return']:.2f}, Trades: {info['num_trades']}")
        
        # Calculate overall metrics
        eval_metrics = {
            'num_episodes': num_episodes,
            'avg_reward': np.mean(episode_rewards),
            'avg_return': np.mean(episode_returns),
            'avg_length': np.mean(episode_lengths),
            'avg_trades': np.mean(episode_trades),
            'std_return': np.std(episode_returns)
        }
        
        return eval_metrics
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the model to disk
        
        Args:
            filepath (Optional[str]): Path to save the model (without extension)
            
        Returns:
            str: Path where the model was saved
        """
        if filepath is None:
            # Create default filepath
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.model_dir, f"{self.name}_{timestamp}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save actor model
        actor_path = f"{filepath}_actor.h5"
        self.actor.save(actor_path)
        
        # Save critic model
        critic_path = f"{filepath}_critic.h5"
        self.critic.save(critic_path)
        
        # Save metadata
        metadata = {
            'name': self.name,
            'version': self.version,
            'params': self.params,
            'metrics': self.metrics,
            'last_update': datetime.now().isoformat()
        }
        
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        
        self.logger.info(f"Model saved to {filepath}")
        return filepath
    
    @classmethod
    def load_model(cls, 
                 filepath: str, 
                 custom_objects: Optional[Dict[str, Any]] = None) -> 'PPOTrader':
        """
        Load a model from disk
        
        Args:
            filepath (str): Path to the saved model (without extension)
            custom_objects (Optional[Dict[str, Any]]): Custom objects to load with the model
            
        Returns:
            PPOTrader: Loaded model
        """
        try:
            # Load metadata
            with open(f"{filepath}_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Create instance
            instance = cls(
                name=metadata['name'],
                version=metadata['version'],
                **metadata['params']
            )
            
            # Load actor model
            instance.actor = keras.models.load_model(f"{filepath}_actor.h5", custom_objects=custom_objects)
            
            # Load critic model
            instance.critic = keras.models.load_model(f"{filepath}_critic.h5", custom_objects=custom_objects)
            
            # Set instance variables
            instance.metrics = metadata['metrics']
            instance.is_trained = True
            instance.last_update = datetime.fromisoformat(metadata['last_update'])
            
            logger.info(f"Model loaded from {filepath}")
            return instance
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot training history
        
        Args:
            figsize (Tuple[int, int]): Figure size
        """
        if 'episode_rewards' not in self.metrics or 'episode_returns' not in self.metrics:
            self.logger.warning("No training history available")
            return
        
        plt.figure(figsize=figsize)
        
        # Plot episode rewards
        plt.subplot(2, 1, 1)
        plt.plot(self.metrics['episode_rewards'])
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # Plot episode returns
        plt.subplot(2, 1, 2)
        plt.plot(self.metrics['episode_returns'])
        plt.title('Portfolio Returns')
        plt.xlabel('Episode')
        plt.ylabel('Return (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show() 