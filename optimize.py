"""
Hyperparameter Optimization Script

This script finds optimal hyperparameters for both ADALINE and LSTM models
using grid search with validation RMSE as the optimization metric.
"""

import numpy as np
from itertools import product
import json
from datetime import datetime

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. LSTM optimization will be skipped.")

from data import fetch_apophis_data, prepare_sequences
from model import Adaline
if TORCH_AVAILABLE:
    from model import LSTMPredictor


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def train_lstm(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.001):
    """
    Train LSTM model using PyTorch.
    
    Returns:
    --------
    List[float]
        Training loss history
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)
    
    return train_losses


def optimize_adaline(X_train, y_train, X_val, y_val, verbose=True):
    """
    Optimize ADALINE hyperparameters using grid search.
    
    Parameters:
    -----------
    X_train, y_train, X_val, y_val : np.ndarray
        Training and validation data
    verbose : bool
        Print progress
        
    Returns:
    --------
    dict
        Best hyperparameters and results
    """
    print("\n" + "=" * 80)
    print("ADALINE Hyperparameter Optimization")
    print("=" * 80)
    
    # Hyperparameter search space (can be expanded)
    learning_rates = [0.005, 0.01, 0.02, 0.05]
    epochs_list = [100, 150, 200]
    
    best_rmse = float('inf')
    best_params = None
    results = []
    
    total_combinations = len(learning_rates) * len(epochs_list)
    current = 0
    
    for lr, epochs in product(learning_rates, epochs_list):
        current += 1
        if verbose:
            print(f"\n[{current}/{total_combinations}] Testing: lr={lr:.4f}, epochs={epochs}")
        
        # Create and train model
        model = Adaline(
            n_features=X_train.shape[1],
            n_outputs=3,
            learning_rate=lr
        )
        model.train(X_train, y_train, epochs=epochs, verbose=False)
        
        # Evaluate on validation set
        val_pred = model.predict(X_val)
        val_rmse = calculate_rmse(y_val, val_pred)
        
        # Train RMSE for reference
        train_pred = model.predict(X_train)
        train_rmse = calculate_rmse(y_train, train_pred)
        
        results.append({
            'learning_rate': lr,
            'epochs': epochs,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'final_mse': model.mse_history[-1] if model.mse_history else None
        })
        
        if verbose:
            print(f"  Train RMSE: {train_rmse:.6f}, Val RMSE: {val_rmse:.6f}")
        
        # Update best if better
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_params = {
                'learning_rate': lr,
                'epochs': epochs,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'final_mse': model.mse_history[-1] if model.mse_history else None
            }
            if verbose:
                print(f"  ⭐ New best! Val RMSE: {val_rmse:.6f}")
    
    print("\n" + "-" * 80)
    print("ADALINE Optimization Complete")
    print("-" * 80)
    print(f"Best Parameters:")
    print(f"  Learning Rate: {best_params['learning_rate']:.4f}")
    print(f"  Epochs: {best_params['epochs']}")
    print(f"  Validation RMSE: {best_params['val_rmse']:.6f}")
    print(f"  Train RMSE: {best_params['train_rmse']:.6f}")
    
    return {
        'best_params': best_params,
        'all_results': results
    }


def optimize_lstm(X_train, y_train, X_val, y_val, verbose=True):
    """
    Optimize LSTM hyperparameters using grid search.
    
    Parameters:
    -----------
    X_train, y_train, X_val, y_val : np.ndarray
        Training and validation data (already prepared as sequences)
    verbose : bool
        Print progress
        
    Returns:
    --------
    dict
        Best hyperparameters and results
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping LSTM optimization.")
        return None
    
    print("\n" + "=" * 80)
    print("LSTM Hyperparameter Optimization")
    print("=" * 80)
    
    # Prepare sequences
    sequence_length = 10
    X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = prepare_sequences(X_val, y_val, sequence_length)
    
    # Hyperparameter search space (reduced for efficiency)
    # Start with a smaller grid, can be expanded
    learning_rates = [0.0005, 0.001, 0.002]
    hidden_dims = [64, 128]
    dropout_rates = [0.0, 0.2]
    epochs_list = [50, 100]
    
    best_rmse = float('inf')
    best_params = None
    results = []
    
    total_combinations = len(learning_rates) * len(hidden_dims) * len(dropout_rates) * len(epochs_list)
    current = 0
    
    for lr, hidden_dim, dropout, epochs in product(learning_rates, hidden_dims, dropout_rates, epochs_list):
        current += 1
        if verbose:
            print(f"\n[{current}/{total_combinations}] Testing: lr={lr:.4f}, hidden={hidden_dim}, dropout={dropout:.2f}, epochs={epochs}")
        
        try:
            # Create model
            model = LSTMPredictor(
                input_dim=X_train.shape[1],
                hidden_dim=hidden_dim,
                output_dim=3,
                num_layers=1,
                dropout=dropout
            )
            
            # Train model
            train_losses = train_lstm(
                model, X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                epochs=epochs,
                batch_size=32,
                lr=lr
            )
            
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val_seq)
                val_pred = model(X_val_tensor).numpy()
            
            val_rmse = calculate_rmse(y_val_seq, val_pred)
            
            # Train RMSE for reference
            model.eval()
            with torch.no_grad():
                X_train_tensor = torch.FloatTensor(X_train_seq)
                train_pred = model(X_train_tensor).numpy()
            
            train_rmse = calculate_rmse(y_train_seq, train_pred)
            final_loss = train_losses[-1] if train_losses else None
            
            results.append({
                'learning_rate': lr,
                'hidden_dim': hidden_dim,
                'dropout': dropout,
                'epochs': epochs,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
                'final_loss': final_loss
            })
            
            if verbose:
                print(f"  Train RMSE: {train_rmse:.6f}, Val RMSE: {val_rmse:.6f}, Final Loss: {final_loss:.6f}" if final_loss else f"  Train RMSE: {train_rmse:.6f}, Val RMSE: {val_rmse:.6f}")
            
            # Update best if better
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_params = {
                    'learning_rate': lr,
                    'hidden_dim': hidden_dim,
                    'dropout': dropout,
                    'epochs': epochs,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'final_loss': final_loss
                }
                if verbose:
                    print(f"  ⭐ New best! Val RMSE: {val_rmse:.6f}")
        
        except Exception as e:
            if verbose:
                print(f"  ❌ Error: {e}")
            continue
    
    if best_params is None:
        print("No successful LSTM training runs.")
        return None
    
    print("\n" + "-" * 80)
    print("LSTM Optimization Complete")
    print("-" * 80)
    print(f"Best Parameters:")
    print(f"  Learning Rate: {best_params['learning_rate']:.4f}")
    print(f"  Hidden Dimensions: {best_params['hidden_dim']}")
    print(f"  Dropout: {best_params['dropout']:.2f}")
    print(f"  Epochs: {best_params['epochs']}")
    print(f"  Validation RMSE: {best_params['val_rmse']:.6f}")
    print(f"  Train RMSE: {best_params['train_rmse']:.6f}")
    
    return {
        'best_params': best_params,
        'all_results': results
    }


def main():
    """Main optimization function."""
    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION")
    print("Finding optimal hyperparameters for ADALINE and LSTM models")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    try:
        X, y, scaler_X, scaler_y, dates = fetch_apophis_data(
            start_date='2026-01-01',
            end_date='2030-01-01'
        )
        print(f"Loaded {len(X)} samples")
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Using synthetic data for demonstration...")
        n_samples = 1461
        t = np.arange(n_samples).reshape(-1, 1)
        X = np.hstack([
            t / 100,
            np.sin(t / 365.25 * 2 * np.pi).reshape(-1, 1) * 0.5,
            np.cos(t / 365.25 * 2 * np.pi).reshape(-1, 1) * 0.5,
            np.sin(t / 365.25 * 4 * np.pi).reshape(-1, 1) * 0.1,
            np.cos(t / 365.25 * 2 * np.pi).reshape(-1, 1) * 0.01,
            -np.sin(t / 365.25 * 2 * np.pi).reshape(-1, 1) * 0.01,
            np.cos(t / 365.25 * 4 * np.pi).reshape(-1, 1) * 0.001,
        ])
        y = X[1:, 1:4]
        X = X[:-1]
        from sklearn.preprocessing import StandardScaler
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_X.fit_transform(X)
        y = scaler_y.fit_transform(y)
    
    # Train/validation split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Optimize ADALINE
    adaline_results = optimize_adaline(X_train, y_train, X_val, y_val, verbose=True)
    
    # Optimize LSTM
    lstm_results = None
    if TORCH_AVAILABLE:
        lstm_results = optimize_lstm(X_train, y_train, X_val, y_val, verbose=True)
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'adaline': adaline_results,
        'lstm': lstm_results
    }
    
    with open('optimization_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nADALINE Optimal Hyperparameters:")
    print(f"  Learning Rate: {adaline_results['best_params']['learning_rate']:.4f}")
    print(f"  Epochs: {adaline_results['best_params']['epochs']}")
    print(f"  Validation RMSE: {adaline_results['best_params']['val_rmse']:.6f}")
    
    if lstm_results:
        print("\nLSTM Optimal Hyperparameters:")
        print(f"  Learning Rate: {lstm_results['best_params']['learning_rate']:.4f}")
        print(f"  Hidden Dimensions: {lstm_results['best_params']['hidden_dim']}")
        print(f"  Dropout: {lstm_results['best_params']['dropout']:.2f}")
        print(f"  Epochs: {lstm_results['best_params']['epochs']}")
        print(f"  Validation RMSE: {lstm_results['best_params']['val_rmse']:.6f}")
    
    print("\nResults saved to: optimization_results.json")
    print("=" * 80)


if __name__ == '__main__':
    main()

