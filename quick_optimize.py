"""
Quick Hyperparameter Optimization Script

This script quickly finds good hyperparameters for both ADALINE and LSTM models
using a smaller search space for faster execution.
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
    """Train LSTM model using PyTorch."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    return []


def optimize_adaline_quick(X_train, y_train, X_val, y_val):
    """Quick ADALINE optimization with smaller search space."""
    print("\n" + "=" * 80)
    print("ADALINE Quick Optimization")
    print("=" * 80)
    
    # Smaller search space
    learning_rates = [0.005, 0.01, 0.02, 0.05]
    epochs_list = [100, 150]
    
    best_rmse = float('inf')
    best_params = None
    
    total = len(learning_rates) * len(epochs_list)
    current = 0
    
    for lr, epochs in product(learning_rates, epochs_list):
        current += 1
        print(f"[{current}/{total}] lr={lr:.3f}, epochs={epochs}...", end=" ")
        
        model = Adaline(n_features=X_train.shape[1], n_outputs=3, learning_rate=lr)
        model.train(X_train, y_train, epochs=epochs, verbose=False)
        
        val_pred = model.predict(X_val)
        val_rmse = calculate_rmse(y_val, val_pred)
        print(f"RMSE: {val_rmse:.6f}")
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_params = {'learning_rate': lr, 'epochs': epochs, 'val_rmse': val_rmse}
            print(f"  ⭐ New best!")
    
    print(f"\nBest ADALINE: lr={best_params['learning_rate']:.4f}, epochs={best_params['epochs']}, RMSE={best_params['val_rmse']:.6f}")
    return best_params


def optimize_lstm_quick(X_train, y_train, X_val, y_val):
    """Quick LSTM optimization with smaller search space."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping LSTM optimization.")
        return None
    
    print("\n" + "=" * 80)
    print("LSTM Quick Optimization")
    print("=" * 80)
    
    # Prepare sequences
    sequence_length = 10
    X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = prepare_sequences(X_val, y_val, sequence_length)
    
    # Smaller search space - focusing on dropout=0.0
    learning_rates = [0.0005, 0.001, 0.002, 0.003]
    hidden_dims = [64, 128]
    dropout_rates = [0.0]  # Focus on no dropout
    epochs_list = [50, 100, 150]
    
    best_rmse = float('inf')
    best_params = None
    
    total = len(learning_rates) * len(hidden_dims) * len(dropout_rates) * len(epochs_list)
    current = 0
    
    for lr, hidden_dim, dropout, epochs in product(learning_rates, hidden_dims, dropout_rates, epochs_list):
        current += 1
        print(f"[{current}/{total}] lr={lr:.4f}, hidden={hidden_dim}, dropout={dropout:.1f}, epochs={epochs}...", end=" ")
        
        try:
            model = LSTMPredictor(
                input_dim=X_train.shape[1],
                hidden_dim=hidden_dim,
                output_dim=3,
                num_layers=1,
                dropout=dropout
            )
            
            train_lstm(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=epochs, lr=lr)
            
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val_seq)
                val_pred = model(X_val_tensor).numpy()
            
            val_rmse = calculate_rmse(y_val_seq, val_pred)
            print(f"RMSE: {val_rmse:.6f}")
            
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_params = {
                    'learning_rate': lr,
                    'hidden_dim': hidden_dim,
                    'dropout': dropout,
                    'epochs': epochs,
                    'val_rmse': val_rmse
                }
                print(f"  ⭐ New best!")
        
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    if best_params:
        print(f"\nBest LSTM: lr={best_params['learning_rate']:.4f}, hidden={best_params['hidden_dim']}, "
              f"dropout={best_params['dropout']:.2f}, epochs={best_params['epochs']}, RMSE={best_params['val_rmse']:.6f}")
    
    return best_params


def main():
    """Main optimization function."""
    print("=" * 80)
    print("QUICK HYPERPARAMETER OPTIMIZATION")
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
        print("Using synthetic data...")
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
    
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
    
    # Optimize
    adaline_best = optimize_adaline_quick(X_train, y_train, X_val, y_val)
    lstm_best = optimize_lstm_quick(X_train, y_train, X_val, y_val) if TORCH_AVAILABLE else None
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'adaline': adaline_best,
        'lstm': lstm_best
    }
    
    with open('quick_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("OPTIMAL HYPERPARAMETERS")
    print("=" * 80)
    print("\nADALINE:")
    print(f"  learning_rate = {adaline_best['learning_rate']:.4f}")
    print(f"  epochs = {adaline_best['epochs']}")
    
    if lstm_best:
        print("\nLSTM:")
        print(f"  learning_rate = {lstm_best['learning_rate']:.4f}")
        print(f"  hidden_dim = {lstm_best['hidden_dim']}")
        print(f"  dropout = {lstm_best['dropout']:.2f}")
        print(f"  epochs = {lstm_best['epochs']}")
    
    print("\nResults saved to: quick_optimization_results.json")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    main()

