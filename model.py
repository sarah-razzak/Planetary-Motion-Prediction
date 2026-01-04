"""
Model Implementations for The Ancient Predictor

This module contains two models representing the evolution of machine learning:
1. Adaline (1960s) - The "Historical" model using Widrow-Hoff Delta Rule
2. LSTM (2026-era) - The "Modern" baseline with sequential memory

The contrast between these models illustrates the "Geometrical Crisis" - how linear
models struggle with the curved trajectories of planetary motion, while recurrent
architectures can capture the non-linear dynamics through hidden state memory.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple


class Adaline:
    """
    Adaptive Linear Neuron (Adaline) - 1960s Architecture
    
    This is the historical model that bridges Gauss's Least Squares (18th century)
    with Rosenblatt's Perceptron. Unlike the Perceptron, Adaline uses a linear
    activation function (identity) rather than a step function, making it suitable
    for regression tasks like predicting continuous 3D positions.
    
    The Widrow-Hoff Delta Rule: w = w + η(y - ŷ)x
    where:
        w: weight vector
        η (eta): learning rate
        y: true label
        ŷ: predicted output (w^T x + b)
        x: input feature vector
    
    The training process demonstrates the "Error Surface" journey - how gradient
    descent navigates the bowl-shaped loss landscape to find the minimum.
    """
    
    def __init__(self, n_features: int, n_outputs: int = 3, learning_rate: float = 0.01):
        """
        Initialize Adaline model.
        
        Parameters:
        -----------
        n_features : int
            Number of input features (Time, X, Y, Z, VX, VY, VZ = 7)
        n_outputs : int
            Number of output dimensions (X, Y, Z = 3)
        learning_rate : float
            Learning rate η (eta) for Widrow-Hoff rule
        """
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        
        # Initialize weights: w (n_features x n_outputs) and bias b (n_outputs)
        # Small random initialization to break symmetry
        self.weights = np.random.randn(n_features, n_outputs) * 0.01
        self.bias = np.random.randn(n_outputs) * 0.01
        
        # Track training history
        self.mse_history = []
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: ŷ = Xw + b
        
        The identity activation function allows continuous outputs, unlike
        the step function used in the Perceptron. This is what makes Adaline
        suitable for regression in 3D space.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features of shape (n_samples, n_features)
        
        Returns:
        --------
        np.ndarray
            Predictions of shape (n_samples, n_outputs)
        """
        return X @ self.weights + self.bias
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, verbose: bool = True):
        """
        Train Adaline using Widrow-Hoff Delta Rule.
        
        This method tracks the Mean Squared Error (MSE) at each epoch to visualize
        the "bottom of the bowl" concept - how the error surface guides the model
        toward the optimal weights.
        
        The Geometrical Crisis: Linear models like Adaline assume a linear relationship
        between inputs and outputs. However, planetary orbits are elliptical (non-linear),
        which means Adaline can only approximate the true trajectory.
        
        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels
        epochs : int
            Number of training epochs
        verbose : bool
            Whether to print training progress
        """
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.predict(X)
            
            # Calculate error: e = y - ŷ
            error = y - y_pred
            
            # Mean Squared Error: MSE = (1/n) Σ(y - ŷ)²
            mse = np.mean(error ** 2)
            self.mse_history.append(mse)
            
            # Widrow-Hoff Delta Rule: w = w + η(y - ŷ)x
            # For batch update, we average over all samples
            # Gradient of MSE w.r.t. weights: ∇w = -(2/n) Σ(y - ŷ)x
            # Update rule: w = w - η * ∇w = w + (2η/n) Σ(y - ŷ)x
            
            # Update weights: w = w + η * (1/n) * X^T * error
            weight_update = self.learning_rate * (1.0 / n_samples) * (X.T @ error)
            self.weights += weight_update
            
            # Update bias: b = b + η * (1/n) * Σ error
            bias_update = self.learning_rate * (1.0 / n_samples) * np.mean(error, axis=0)
            self.bias += bias_update
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, MSE: {mse:.6f}")
        
        if verbose:
            print(f"\nTraining complete. Final MSE: {self.mse_history[-1]:.6f}")
            print(f"Error reduction: {self.mse_history[0]:.6f} → {self.mse_history[-1]:.6f}")
    
    def get_error_history(self) -> List[float]:
        """Return the MSE history to visualize the error surface journey."""
        return self.mse_history


class LSTMPredictor(nn.Module):
    """
    Long Short-Term Memory (LSTM) Network - 2026-era Architecture
    
    The LSTM solves the "Geometrical Crisis" that plagues linear models like Adaline.
    Through its hidden states and gating mechanisms, the LSTM maintains memory of
    previous positions and velocities, allowing it to capture the non-linear dynamics
    of elliptical orbits.
    
    Key Innovation: Hidden States
    -----------------------------
    Unlike Adaline, which treats each time step independently, the LSTM maintains
    a hidden state h_t that encodes information from all previous time steps. This
    allows it to learn patterns like:
    - Orbital periodicity
    - Velocity-position relationships across time
    - Acceleration patterns that depend on distance from the sun
    
    The LSTM's ability to model these non-linear relationships makes it superior
    for predicting curved trajectories in 3D space.
    """
    
    def __init__(self, input_dim: int = 7, hidden_dim: int = 64, output_dim: int = 3, num_layers: int = 1):
        """
        Initialize LSTM model.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features per time step
        hidden_dim : int
            Dimension of hidden state (memory capacity)
        output_dim : int
            Number of output dimensions (X, Y, Z)
        num_layers : int
            Number of LSTM layers
        """
        super(LSTMPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer: processes sequences and maintains hidden state
        # The hidden state h_t captures the "memory" of previous positions and velocities
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Linear head: maps hidden state to output position
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        The LSTM processes the input sequence and produces a hidden state for each
        time step. We use the final hidden state to predict the next position.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input sequence of shape (batch_size, sequence_length, input_dim)
        
        Returns:
        --------
        torch.Tensor
            Predictions of shape (batch_size, output_dim)
        """
        # LSTM forward pass
        # lstm_out: (batch_size, sequence_length, hidden_dim)
        # hidden: tuple of (h_n, c_n) where h_n is the final hidden state
        lstm_out, hidden = self.lstm(x)
        
        # Use the final hidden state (last time step) for prediction
        # lstm_out[:, -1, :] extracts the hidden state at the last time step
        final_hidden = lstm_out[:, -1, :]
        
        # Linear transformation to output space
        output = self.fc(final_hidden)
        
        return output

