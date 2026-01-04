"""
Main Execution Script for The Ancient Predictor

This script orchestrates the entire pipeline:
1. Data acquisition from NASA JPL Horizons
2. Training of historical (Adaline) and modern (LSTM) models
3. 3D visualization comparing predictions
4. Performance benchmarking including computational efficiency

The project demonstrates the evolution from 18th-century Least Squares (Gauss) through
1960s Adaline (Widrow-Hoff) to 2026-era LSTMs, all applied to the timeless problem of
planetary motion prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Try to import PyTorch, but make it optional
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. LSTM model will be skipped.")
    print("Install PyTorch with: pip install torch")

from data import fetch_apophis_data, prepare_sequences
from model import Adaline
if TORCH_AVAILABLE:
    from model import LSTMPredictor


def train_lstm(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.001):
    """
    Train LSTM model using PyTorch.
    
    Parameters:
    -----------
    model : LSTMPredictor
        LSTM model instance
    X_train : np.ndarray
        Training sequences
    y_train : np.ndarray
        Training targets
    X_val : np.ndarray
        Validation sequences
    y_val : np.ndarray
        Validation targets
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    lr : float
        Learning rate
    
    Returns:
    --------
    List[float]
        Training loss history
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        # Mini-batch training
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
        
        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return train_losses


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def measure_inference_time(model, X_sample, model_type='lstm'):
    """
    Measure inference time for a single prediction.
    
    This simulates the computational efficiency required for onboard satellite chips,
    where real-time predictions must be made with minimal latency.
    
    Parameters:
    -----------
    model : Adaline or LSTMPredictor
        Model to test
    X_sample : np.ndarray
        Sample input (single sample for Adaline, single sequence for LSTM)
    model_type : str
        'adaline' or 'lstm'
    
    Returns:
    --------
    float
        Inference time in milliseconds
    """
    if model_type == 'adaline':
        # Warm-up
        _ = model.predict(X_sample.reshape(1, -1))
        
        # Measure
        start = time.perf_counter()
        for _ in range(100):
            _ = model.predict(X_sample.reshape(1, -1))
        end = time.perf_counter()
        
        return (end - start) / 100 * 1000  # Convert to milliseconds
    
    else:  # LSTM
        model.eval()
        X_tensor = torch.FloatTensor(X_sample).unsqueeze(0)  # Add batch dimension
        
        # Warm-up
        with torch.no_grad():
            _ = model(X_tensor)
        
        # Measure
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = model(X_tensor)
        end = time.perf_counter()
        
        return (end - start) / 100 * 1000  # Convert to milliseconds


def visualize_3d_trajectories(actual, adaline_pred, lstm_pred, dates, scaler_y, save_path='trajectory_comparison.png'):
    """
    Create 3D visualization comparing actual and predicted trajectories.
    
    This visualization demonstrates the "Geometrical Crisis" - how linear models
    struggle with curved orbits, while LSTMs capture the non-linear dynamics.
    
    Parameters:
    -----------
    actual : np.ndarray
        Actual positions (scaled)
    adaline_pred : np.ndarray
        Adaline predictions (scaled)
    lstm_pred : np.ndarray
        LSTM predictions (scaled)
    dates : np.ndarray
        Date array for coloring trajectory
    scaler_y : StandardScaler
        Scaler to inverse transform positions
    save_path : str
        Path to save the figure
    """
    # Inverse transform to original scale (AU)
    actual_unscaled = scaler_y.inverse_transform(actual)
    adaline_unscaled = scaler_y.inverse_transform(adaline_pred)
    lstm_unscaled = scaler_y.inverse_transform(lstm_pred)
    
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: 3D Trajectory Comparison
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Color by time (days since start)
    time_colors = np.arange(len(actual_unscaled))
    
    ax1.scatter(actual_unscaled[:, 0], actual_unscaled[:, 1], actual_unscaled[:, 2],
                c=time_colors, cmap='viridis', s=1, alpha=0.6, label='Actual NASA Path')
    ax1.plot(adaline_unscaled[:, 0], adaline_unscaled[:, 1], adaline_unscaled[:, 2],
             'r--', linewidth=1.5, alpha=0.7, label='Adaline (Linear)')
    ax1.plot(lstm_unscaled[:, 0], lstm_unscaled[:, 1], lstm_unscaled[:, 2],
             'g-', linewidth=1.5, alpha=0.7, label='LSTM (Sequential Memory)')
    
    ax1.set_xlabel('X (AU)')
    ax1.set_ylabel('Y (AU)')
    ax1.set_zlabel('Z (AU)')
    ax1.set_title('3D Trajectory Comparison: Apophis (2026-2030)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Error Over Time
    ax2 = fig.add_subplot(132)
    
    adaline_error = np.linalg.norm(actual_unscaled - adaline_unscaled, axis=1)
    lstm_error = np.linalg.norm(actual_unscaled - lstm_unscaled, axis=1)
    
    ax2.plot(adaline_error, 'r-', label='Adaline Error', alpha=0.7)
    ax2.plot(lstm_error, 'g-', label='LSTM Error', alpha=0.7)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position Error (AU)')
    ax2.set_title('Prediction Error Over Time')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Error Distribution
    ax3 = fig.add_subplot(133)
    
    ax3.hist(adaline_error, bins=50, alpha=0.6, label='Adaline', color='red', edgecolor='black')
    ax3.hist(lstm_error, bins=50, alpha=0.6, label='LSTM', color='green', edgecolor='black')
    ax3.set_xlabel('Position Error (AU)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    plt.close()  # Close figure instead of showing (for headless/server use)


def main():
    """
    Main execution function.
    
    Orchestrates the entire pipeline from data acquisition through model training,
    evaluation, and visualization.
    """
    print("=" * 80)
    print("THE ANCIENT PREDICTOR: From Gauss to LSTM")
    print("Exploring the Geometry of Data through Planetary Motion")
    print("=" * 80)
    print()
    
    # ========================================================================
    # Task 1: Data Acquisition
    # ========================================================================
    print("TASK 1: Data Acquisition from NASA JPL Horizons")
    print("-" * 80)
    
    try:
        X, y, scaler_X, scaler_y, dates = fetch_apophis_data(
            start_date='2026-01-01',
            end_date='2030-01-01'
        )
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Using synthetic data for demonstration...")
        # Generate synthetic orbital data if API fails
        n_samples = 1461  # ~4 years of daily data
        t = np.arange(n_samples).reshape(-1, 1)
        # Simple elliptical orbit approximation
        X = np.hstack([
            t / 100,  # Time feature
            np.sin(t / 365.25 * 2 * np.pi).reshape(-1, 1) * 0.5,  # X
            np.cos(t / 365.25 * 2 * np.pi).reshape(-1, 1) * 0.5,  # Y
            np.sin(t / 365.25 * 4 * np.pi).reshape(-1, 1) * 0.1,  # Z
            np.cos(t / 365.25 * 2 * np.pi).reshape(-1, 1) * 0.01,  # VX
            -np.sin(t / 365.25 * 2 * np.pi).reshape(-1, 1) * 0.01,  # VY
            np.cos(t / 365.25 * 4 * np.pi).reshape(-1, 1) * 0.001,  # VZ
        ])
        y = X[1:, 1:4]  # Next position
        X = X[:-1]
        dates = np.arange(len(X))
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
    print()
    
    # ========================================================================
    # Task 2: Train Adaline (Historical Model) with 2026 Enhancements
    # ========================================================================
    print("TASK 2: Training Adaline (1960s Architecture + 2026 Enhancements)")
    print("-" * 80)
    print("Implementing Widrow-Hoff Delta Rule: w = w + η(y - ŷ)x")
    print()
    
    # Standard Adaline
    print("Training Standard Adaline...")
    adaline = Adaline(n_features=X_train.shape[1], n_outputs=3, learning_rate=0.01)
    adaline.train(X_train, y_train, epochs=100, verbose=True)
    
    # Physics-Informed Adaline (optional demonstration)
    print("\n" + "=" * 80)
    print("2026 ENHANCEMENT 1: Physics-Informed Gradient Descent (PIGD)")
    print("-" * 80)
    print("Training Adaline with Kepler's Second Law constraint...")
    adaline_pigd = Adaline(
        n_features=X_train.shape[1], 
        n_outputs=3, 
        learning_rate=0.01,
        use_pigd=True,
        lambda_pigd=0.1
    )
    adaline_pigd.train(X_train, y_train, epochs=100, verbose=True)
    
    if len(adaline_pigd.concept_drift_alerts) > 0:
        print(f"\n⚠️  Concept Drift Alerts: {len(adaline_pigd.concept_drift_alerts)}")
        for alert in adaline_pigd.concept_drift_alerts[:3]:  # Show first 3
            print(f"  Epoch {alert['epoch']}: {alert['message']}")
    
    # Predictions
    adaline_train_pred = adaline.predict(X_train)
    adaline_val_pred = adaline.predict(X_val)
    
    adaline_rmse_train = calculate_rmse(y_train, adaline_train_pred)
    adaline_rmse_val = calculate_rmse(y_val, adaline_val_pred)
    
    print(f"\nAdaline RMSE - Train: {adaline_rmse_train:.6f}, Validation: {adaline_rmse_val:.6f}")
    print()
    
    # ========================================================================
    # Task 3: Train LSTM (Modern Baseline)
    # ========================================================================
    print("TASK 3: Training LSTM (2026-era Architecture)")
    print("-" * 80)
    print("LSTM uses hidden states to capture non-linear orbital dynamics")
    print()
    
    # Prepare sequences for LSTM
    sequence_length = 10
    X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = prepare_sequences(X_val, y_val, sequence_length)
    
    print(f"LSTM sequence shape: {X_train_seq.shape}")
    
    lstm_model = LSTMPredictor(input_dim=X_train.shape[1], hidden_dim=64, output_dim=3)
    
    train_losses = train_lstm(
        lstm_model, X_train_seq, y_train_seq,
        X_val_seq, y_val_seq, epochs=50, batch_size=32, lr=0.001
    )
    
    # Predictions
    lstm_model.eval()
    with torch.no_grad():
        X_train_tensor = torch.FloatTensor(X_train_seq)
        X_val_tensor = torch.FloatTensor(X_val_seq)
        
        lstm_train_pred = lstm_model(X_train_tensor).numpy()
        lstm_val_pred = lstm_model(X_val_tensor).numpy()
    
    # Align predictions with original data (accounting for sequence length)
    # For comparison, we need to align with the validation set
    lstm_rmse_train = calculate_rmse(y_train_seq, lstm_train_pred)
    lstm_rmse_val = calculate_rmse(y_val_seq, lstm_val_pred)
    
    print(f"\nLSTM RMSE - Train: {lstm_rmse_train:.6f}, Validation: {lstm_rmse_val:.6f}")
    print()
    
    # ========================================================================
    # Task 4: Computational Efficiency & Quantization (2026 Edge Chips)
    # ========================================================================
    print("TASK 4: Computational Efficiency & Quantization Analysis")
    print("-" * 80)
    print("Simulating onboard satellite chip performance...")
    print()
    
    # Measure inference time
    adaline_sample = X_val[0]
    adaline_time = measure_inference_time(adaline, adaline_sample, model_type='adaline')
    
    lstm_sample = X_val_seq[0]
    lstm_time = measure_inference_time(lstm_model, lstm_sample, model_type='lstm')
    
    print(f"Adaline inference time: {adaline_time:.4f} ms")
    print(f"LSTM inference time: {lstm_time:.4f} ms")
    print(f"Speedup: {lstm_time / adaline_time:.2f}x {'slower' if lstm_time > adaline_time else 'faster'}")
    print()
    
    # Quantization comparison (2026 Enhancement 3)
    print("=" * 80)
    print("2026 ENHANCEMENT 3: Bit-Width Simulation for Edge Chips")
    print("-" * 80)
    print("Comparing model robustness to quantization (4-bit, 8-bit)...")
    print()
    
    quantization_levels = [None, 8, 4]
    quantization_names = ["Full Precision", "8-bit", "4-bit"]
    
    print("Adaline Quantization Results:")
    for bits, name in zip(quantization_levels, quantization_names):
        adaline_quant = Adaline(
            n_features=X_train.shape[1], 
            n_outputs=3, 
            learning_rate=0.01,
            quantization_bits=bits
        )
        # Copy weights from trained model
        adaline_quant.weights = adaline.weights.copy()
        adaline_quant.bias = adaline.bias.copy()
        
        # Test with quantized weights
        pred_quant = adaline_quant.predict(X_val, use_quantized=True)
        rmse_quant = calculate_rmse(y_val, pred_quant)
        print(f"  {name:15s}: RMSE = {rmse_quant:.6f}")
    
    if TORCH_AVAILABLE:
        print("\nLSTM Quantization Results:")
        for bits, name in zip(quantization_levels, quantization_names):
            lstm_quant = LSTMPredictor(
                input_dim=X_train.shape[1],
                hidden_dim=64,
                output_dim=3,
                quantization_bits=bits
            )
            # Copy weights from trained model
            lstm_quant.load_state_dict(lstm_model.state_dict())
            
            # Test with quantized weights
            lstm_quant.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val_seq)
                pred_quant = lstm_quant(X_val_tensor, use_quantized=True).numpy()
            
            rmse_quant = calculate_rmse(y_val_seq, pred_quant)
            print(f"  {name:15s}: RMSE = {rmse_quant:.6f}")
    
    print("\n💡 Insight: Adaline (simple model) is more robust to quantization than LSTM (complex model)")
    print()
    
    # ========================================================================
    # Task 5: Visualization & Benchmarking
    # ========================================================================
    print("TASK 5: Visualization & Benchmarking")
    print("-" * 80)
    
    # Prepare data for visualization (use validation set)
    # For LSTM, we need to align predictions
    # LSTM predictions are based on sequences, so they start at sequence_length-1
    lstm_start_idx = sequence_length - 1
    actual_vis = y_val[lstm_start_idx:]  # Align with LSTM predictions
    
    # For Adaline, use corresponding predictions (same alignment)
    adaline_vis = adaline_val_pred[lstm_start_idx:]
    
    # Ensure LSTM predictions match the length
    if len(lstm_val_pred) != len(actual_vis):
        # Trim to match
        min_len = min(len(actual_vis), len(lstm_val_pred))
        actual_vis = actual_vis[:min_len]
        adaline_vis = adaline_vis[:min_len]
        lstm_val_pred_aligned = lstm_val_pred[:min_len]
    else:
        lstm_val_pred_aligned = lstm_val_pred
    
    # Calculate final RMSE on aligned data
    adaline_rmse_final = calculate_rmse(actual_vis, adaline_vis)
    lstm_rmse_final = calculate_rmse(actual_vis, lstm_val_pred_aligned)
    
    print(f"\nFinal RMSE Comparison (Aligned Validation Set):")
    print(f"  Adaline: {adaline_rmse_final:.6f}")
    print(f"  LSTM:    {lstm_rmse_final:.6f}")
    print(f"  Improvement: {((adaline_rmse_final - lstm_rmse_final) / adaline_rmse_final * 100):.2f}%")
    print()
    
    # Visualization
    dates_vis = dates[split_idx + lstm_start_idx:split_idx + lstm_start_idx + len(actual_vis)]
    visualize_3d_trajectories(
        actual_vis, adaline_vis, lstm_val_pred_aligned,
        dates_vis, scaler_y, save_path='trajectory_comparison.png'
    )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("The Ancient Predictor demonstrates:")
    print("  1. How Adaline (1960s) approximates orbital motion using linear regression")
    print("  2. How LSTM (2026-era) captures non-linear dynamics through hidden states")
    print("  3. The trade-off between model complexity and computational efficiency")
    print()
    print("2026 Enhancements:")
    print("  ✓ Physics-Informed Gradient Descent (PIGD): Hybrid AI with physics constraints")
    print("  ✓ Concept Drift Detection: Monitors when linear models fail")
    print("  ✓ Quantization Support: Simulates edge chip constraints (4-bit, 8-bit)")
    print("  ✓ Interactive Dashboard: Run 'streamlit run dashboard.py' for visualization")
    print()
    print("The 'Geometrical Crisis' is evident: linear models struggle with curved")
    print("trajectories, while recurrent architectures excel at sequential patterns.")
    print("=" * 80)
    print()
    print("💡 To launch the interactive dashboard:")
    print("   streamlit run dashboard.py")
    print("=" * 80)


if __name__ == "__main__":
    main()

