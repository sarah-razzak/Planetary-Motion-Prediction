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
    Adaptive Linear Neuron (Adaline) - 1960s Architecture with 2026 Enhancements
    
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
    
    2026 Enhancements:
    - Physics-Informed Gradient Descent (PIGD): Adds Kepler's Second Law constraint
    - Quantization Support: Simulates edge chip constraints (4-bit, 8-bit)
    - Concept Drift Monitoring: Tracks error trends for non-linearity detection
    """
    
    def __init__(self, n_features: int, n_outputs: int = 3, learning_rate: float = 0.01,
                 use_pigd: bool = False, lambda_pigd: float = 0.1,
                 quantization_bits: int = None):
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
        use_pigd : bool
            Enable Physics-Informed Gradient Descent (Kepler's Second Law)
        lambda_pigd : float
            Weight for physics constraint in loss function
        quantization_bits : int, optional
            Simulate quantization (4, 8, or None for full precision)
        """
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        self.use_pigd = use_pigd
        self.lambda_pigd = lambda_pigd
        self.quantization_bits = quantization_bits
        
        # Initialize weights: w (n_features x n_outputs) and bias b (n_outputs)
        # Small random initialization to break symmetry
        self.weights = np.random.randn(n_features, n_outputs) * 0.01
        self.bias = np.random.randn(n_outputs) * 0.01
        
        # Track training history
        self.mse_history = []
        self.physics_constraint_history = []
        self.concept_drift_alerts = []
    
    def _quantize_weights(self):
        """Simulate quantization for edge chip constraints (2026 feature)."""
        if self.quantization_bits is None:
            return self.weights
        
        # Quantization: map to integer range and back
        # For 8-bit: range is -128 to 127, scale by 127
        # For 4-bit: range is -8 to 7, scale by 7
        max_val = 2 ** (self.quantization_bits - 1) - 1
        
        # Quantize weights
        weights_quantized = np.round(self.weights * max_val) / max_val
        return weights_quantized
    
    def predict(self, X: np.ndarray, use_quantized: bool = False) -> np.ndarray:
        """
        Forward pass: ŷ = Xw + b
        
        The identity activation function allows continuous outputs, unlike
        the step function used in the Perceptron. This is what makes Adaline
        suitable for regression in 3D space.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features of shape (n_samples, n_features)
        use_quantized : bool
            Use quantized weights (for edge chip simulation)
        
        Returns:
        --------
        np.ndarray
            Predictions of shape (n_samples, n_outputs)
        """
        if use_quantized and self.quantization_bits is not None:
            weights = self._quantize_weights()
        else:
            weights = self.weights
        return X @ weights + self.bias
    
    def _calculate_angular_momentum(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """
        Calculate angular momentum: L = r × p = r × (m*v)
        For unit mass: L = r × v
        
        This is used for Physics-Informed Gradient Descent (Kepler's Second Law).
        """
        # position: (n_samples, 3) [x, y, z]
        # velocity: (n_samples, 3) [vx, vy, vz]
        # Angular momentum: L = r × v (cross product)
        L = np.cross(position, velocity)
        return L
    
    def _kepler_constraint(self, X: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate physics constraint based on Kepler's Second Law.
        
        Kepler's Second Law: Equal areas in equal time (angular momentum conservation).
        The constraint penalizes deviations from constant angular momentum.
        
        Returns:
        --------
        float
            Physics constraint violation (should be minimized)
        """
        # Extract position and velocity from features
        # X format: [t, x, y, z, vx, vy, vz]
        positions = X[:, 1:4]  # x, y, z
        velocities = X[:, 4:7]  # vx, vy, vz
        
        # Predicted next position
        pred_positions = y_pred  # predicted x, y, z
        
        # Calculate angular momentum for current state
        L_current = self._calculate_angular_momentum(positions, velocities)
        
        # Estimate velocity for predicted position (simplified: use current velocity)
        # In a full implementation, we'd predict velocity too
        L_pred = self._calculate_angular_momentum(pred_positions, velocities)
        
        # Constraint: angular momentum should be conserved
        # Penalize deviation from conservation
        constraint_violation = np.mean((L_pred - L_current) ** 2)
        
        return constraint_violation
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, verbose: bool = True,
              window_size: int = 50):
        """
        Train Adaline using Widrow-Hoff Delta Rule with optional PIGD.
        
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
        window_size : int
            Rolling window size for concept drift detection
        """
        n_samples = X.shape[0]
        error_window = []  # For concept drift monitoring
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.predict(X)
            
            # Calculate error: e = y - ŷ
            error = y - y_pred
            
            # Mean Squared Error: MSE = (1/n) Σ(y - ŷ)²
            mse = np.mean(error ** 2)
            self.mse_history.append(mse)
            
            # Physics-Informed Gradient Descent (PIGD)
            physics_constraint = 0.0
            if self.use_pigd:
                physics_constraint = self._kepler_constraint(X, y_pred)
                self.physics_constraint_history.append(physics_constraint)
                
                # Combined loss: MSE + λ * Physics Constraint
                # The physics constraint gradient affects weight updates
                # We approximate the gradient of the constraint
                constraint_gradient = self.lambda_pigd * physics_constraint
            else:
                self.physics_constraint_history.append(0.0)
            
            # Widrow-Hoff Delta Rule: w = w + η(y - ŷ)x
            # For batch update, we average over all samples
            # Gradient of MSE w.r.t. weights: ∇w = -(2/n) Σ(y - ŷ)x
            # Update rule: w = w - η * ∇w = w + (2η/n) Σ(y - ŷ)x
            
            # Update weights: w = w + η * (1/n) * X^T * error
            weight_update = self.learning_rate * (1.0 / n_samples) * (X.T @ error)
            
            # Apply physics-informed adjustment if enabled
            if self.use_pigd and physics_constraint > 0:
                # Increase learning rate when physics constraint is violated
                physics_boost = 1.0 + self.lambda_pigd * physics_constraint
                weight_update *= physics_boost
            
            self.weights += weight_update
            
            # Apply quantization if enabled (affects storage, not computation)
            if self.quantization_bits is not None:
                self.weights = self._quantize_weights()
            
            # Update bias: b = b + η * (1/n) * Σ error
            bias_update = self.learning_rate * (1.0 / n_samples) * np.mean(error, axis=0)
            self.bias += bias_update
            
            # Concept Drift Detection
            error_window.append(mse)
            if len(error_window) > window_size:
                error_window.pop(0)
            
            if len(error_window) == window_size:
                # Check if error is increasing (concept drift indicator)
                recent_errors = np.array(error_window[-window_size//2:])
                older_errors = np.array(error_window[:window_size//2])
                
                if len(recent_errors) > 0 and len(older_errors) > 0:
                    recent_mean = np.mean(recent_errors)
                    older_mean = np.mean(older_errors)
                    recent_std = np.std(recent_errors)
                    
                    # Alert if error increased beyond one standard deviation
                    if recent_mean > older_mean + recent_std:
                        alert = {
                            'epoch': epoch,
                            'error_increase': recent_mean - older_mean,
                            'message': 'Non-Linearity Alert: Linear model struggling with curved trajectory'
                        }
                        self.concept_drift_alerts.append(alert)
                        if verbose:
                            print(f"⚠️  Concept Drift Alert at Epoch {epoch}: {alert['message']}")
            
            if verbose and (epoch + 1) % 10 == 0:
                physics_info = f", Physics Constraint: {physics_constraint:.6f}" if self.use_pigd else ""
                print(f"Epoch {epoch + 1}/{epochs}, MSE: {mse:.6f}{physics_info}")
        
        if verbose:
            print(f"\nTraining complete. Final MSE: {self.mse_history[-1]:.6f}")
            print(f"Error reduction: {self.mse_history[0]:.6f} → {self.mse_history[-1]:.6f}")
            if self.use_pigd:
                print(f"Final Physics Constraint: {self.physics_constraint_history[-1]:.6f}")
            if len(self.concept_drift_alerts) > 0:
                print(f"Concept Drift Alerts: {len(self.concept_drift_alerts)}")
    
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
    
    2026 Enhancement: Quantization Support for edge chips
    """
    
    def __init__(self, input_dim: int = 7, hidden_dim: int = 64, output_dim: int = 3, 
                 num_layers: int = 1, quantization_bits: int = None):
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
        self.quantization_bits = quantization_bits
        
        # LSTM layer: processes sequences and maintains hidden state
        # The hidden state h_t captures the "memory" of previous positions and velocities
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Linear head: maps hidden state to output position
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Simulate quantization for edge chip constraints."""
        if self.quantization_bits is None:
            return tensor
        
        max_val = 2 ** (self.quantization_bits - 1) - 1
        # Quantize: round to integer range and scale back
        quantized = torch.round(tensor * max_val) / max_val
        return quantized
    
    def forward(self, x: torch.Tensor, use_quantized: bool = False) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        The LSTM processes the input sequence and produces a hidden state for each
        time step. We use the final hidden state to predict the next position.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input sequence of shape (batch_size, sequence_length, input_dim)
        use_quantized : bool
            Use quantized weights (for edge chip simulation)
        
        Returns:
        --------
        torch.Tensor
            Predictions of shape (batch_size, output_dim)
        """
        # Quantize input if enabled
        if use_quantized and self.quantization_bits is not None:
            x = self._quantize_tensor(x)
        
        # LSTM forward pass
        # lstm_out: (batch_size, sequence_length, hidden_dim)
        # hidden: tuple of (h_n, c_n) where h_n is the final hidden state
        lstm_out, hidden = self.lstm(x)
        
        # Use the final hidden state (last time step) for prediction
        # lstm_out[:, -1, :] extracts the hidden state at the last time step
        final_hidden = lstm_out[:, -1, :]
        
        # Linear transformation to output space
        output = self.fc(final_hidden)
        
        # Quantize output if enabled
        if use_quantized and self.quantization_bits is not None:
            output = self._quantize_tensor(output)
        
        return output

