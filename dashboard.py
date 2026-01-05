"""
Interactive Error Surface Dashboard

A comprehensive Streamlit dashboard for visualizing the error surface, training dynamics,
and comparing Linear Regression vs LSTM models. All visualizations integrated from main.py.
"""

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit or Plotly not available. Install with: pip install streamlit plotly")

import numpy as np

from data import fetch_apophis_data, prepare_sequences
from model import Adaline

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from model import LSTMPredictor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if not STREAMLIT_AVAILABLE:
    print("\n" + "="*80)
    print("ERROR: Streamlit is not installed.")
    print("="*80)
    print("To use the dashboard, install dependencies:")
    print("  pip install streamlit plotly")
    print("\nThen run:")
    print("  streamlit run dashboard.py")
    print("="*80)
    exit(1)

# Page config
st.set_page_config(
    page_title="Linear Regression vs LSTM - Interactive Dashboard",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌌 Linear Regression vs LSTM: Interactive Dashboard")
st.markdown("### Compare Linear Regression vs LSTM with real NASA asteroid data")

# Sidebar controls
st.sidebar.header("⚙️ Training Parameters")

learning_rate = st.sidebar.slider(
    "Learning Rate (η)",
    min_value=0.001,
    max_value=0.5,
    value=0.05,  # Optimal value from hyperparameter optimization
    step=0.001,
    help="Controls the step size in gradient descent. Too high = bouncing out of bowl!"
)

use_pigd = st.sidebar.checkbox(
    "Enable Physics-Informed Gradient Descent (PIGD)",
    value=False,
    help="Adds Kepler's Second Law constraint to the loss function"
)

lambda_pigd = st.sidebar.slider(
    "PIGD Lambda (λ)",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01,
    disabled=not use_pigd,
    help="Weight for physics constraint"
)

quantization_bits = st.sidebar.selectbox(
    "Quantization (Edge Chip Simulation)",
    options=[None, 8, 4],
    format_func=lambda x: "Full Precision" if x is None else f"{x}-bit",
    help="Simulate edge chip constraints"
)

epochs = st.sidebar.slider(
    "Training Epochs",
    min_value=10,
    max_value=200,
    value=150,  # Optimal value from hyperparameter optimization
    step=10
)

train_lstm = st.sidebar.checkbox(
    "Train LSTM Model",
    value=True,
    help="Compare with modern LSTM architecture"
)

lstm_epochs = st.sidebar.slider(
    "LSTM Epochs",
    min_value=10,
    max_value=200,  # Increased max to allow 150
    value=150,  # Optimal value from hyperparameter optimization
    step=10,
    disabled=not train_lstm
)

lstm_lr = st.sidebar.slider(
    "LSTM Learning Rate",
    min_value=0.0001,
    max_value=0.01,
    value=0.0005,  # Optimal value from hyperparameter optimization
    step=0.0001,
    format="%.4f",
    disabled=not train_lstm,
    help="Lower learning rate can help prevent overfitting"
)

# Load data
@st.cache_data
def load_data():
    """Load and cache the asteroid data."""
    try:
        # Use same date range as optimization for consistency
        X, y, scaler_X, scaler_y, dates = fetch_apophis_data('2026-01-01', '2030-01-01')
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        return X_train, y_train, X_val, y_val, scaler_X, scaler_y, dates
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None, None

X_train, y_train, X_val, y_val, scaler_X, scaler_y, dates = load_data()

def create_scheduler_safe(optimizer):
    """Safely create a learning rate scheduler, returning None if it fails."""
    try:
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=False
        )
    except:
        return None

def train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=150, batch_size=32, lr=0.0005):
    """Train LSTM model with progress tracking."""
    # Ensure model is in training mode and has parameters
    model.train()
    
    # Check if model has parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if param_count == 0:
        raise ValueError("Model has no trainable parameters")
    
    criterion = nn.MSELoss()
    
    # Create optimizer with error handling
    try:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    except Exception as e:
        st.error(f"Failed to create optimizer: {e}")
        raise
    
    # Learning rate scheduler to reduce LR when validation loss plateaus
    # Make scheduler optional - if creation fails, continue without it
    scheduler = create_scheduler_safe(optimizer)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 15  # Early stopping patience
    patience_counter = 0
    best_model_state = None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
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
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Early stopping: save best model and check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            status_text.text(f"Early stopping at epoch {epoch + 1} (best val loss: {best_val_loss:.6f})")
            model.load_state_dict(best_model_state)
            break
    
    progress_bar.empty()
    status_text.empty()
    return train_losses

if X_train is not None:
    # Train models
    if st.sidebar.button("🚀 Train Models", type="primary"):
        # Train Linear Regression
        with st.spinner("Training Linear Regression model..."):
            adaline = Adaline(
                n_features=X_train.shape[1],
                n_outputs=3,
                learning_rate=learning_rate,
                use_pigd=use_pigd,
                lambda_pigd=lambda_pigd,
                quantization_bits=quantization_bits
            )
            
            adaline.train(X_train, y_train, epochs=epochs, verbose=False)
            
            # Store in session state
            st.session_state['adaline'] = adaline
            st.session_state['adaline_pred'] = adaline.predict(X_val)
            st.session_state['mse_history'] = adaline.mse_history
            st.session_state['physics_history'] = adaline.physics_constraint_history
            st.session_state['alerts'] = adaline.concept_drift_alerts
        
        # Train LSTM if requested
        if train_lstm and TORCH_AVAILABLE:
            with st.spinner("Training LSTM model..."):
                try:
                    sequence_length = 10  # Optimal from hyperparameter optimization
                    X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, sequence_length)
                    X_val_seq, y_val_seq = prepare_sequences(X_val, y_val, sequence_length)
                    
                    # Validate sequences
                    if len(X_train_seq) == 0 or len(X_val_seq) == 0:
                        st.error("Not enough data to create sequences. Need at least sequence_length samples.")
                        st.stop()
                    
                    lstm_model = LSTMPredictor(
                        input_dim=X_train.shape[1],
                        hidden_dim=128,  # Optimal value from hyperparameter optimization
                        output_dim=3,
                        num_layers=2,  # Deeper network for better performance
                        dropout=0.0,  # Optimal value from hyperparameter optimization
                        quantization_bits=quantization_bits
                    )
                    
                    # Initialize model with a dummy forward pass to ensure it's ready
                    with torch.no_grad():
                        dummy_input = torch.FloatTensor(X_train_seq[:1])
                        _ = lstm_model(dummy_input)
                    
                    lstm_losses = train_lstm_model(
                        lstm_model, X_train_seq, y_train_seq,
                        X_val_seq, y_val_seq, epochs=lstm_epochs, lr=lstm_lr
                    )
                except Exception as e:
                    st.error(f"Error training LSTM model: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()
                
                # Get predictions
                lstm_model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val_seq)
                    lstm_pred = lstm_model(X_val_tensor).numpy()
                
                # Debug: Check shapes and values
                # st.write(f"Debug - y_val_seq shape: {y_val_seq.shape}, lstm_pred shape: {lstm_pred.shape}")
                # st.write(f"Debug - y_val_seq range: [{y_val_seq.min():.3f}, {y_val_seq.max():.3f}]")
                # st.write(f"Debug - lstm_pred range: [{lstm_pred.min():.3f}, {lstm_pred.max():.3f}]")
                
                st.session_state['lstm'] = lstm_model
                st.session_state['lstm_pred'] = lstm_pred
                st.session_state['lstm_losses'] = lstm_losses
                st.session_state['y_val_seq'] = y_val_seq
                st.session_state['X_val_seq'] = X_val_seq
                st.session_state['sequence_length'] = sequence_length
    
    # Display results if model is trained
    if 'adaline' in st.session_state and 'adaline_pred' in st.session_state:
        adaline = st.session_state['adaline']
        adaline_pred = st.session_state['adaline_pred']
        mse_history = st.session_state.get('mse_history', [])
        physics_history = st.session_state.get('physics_history', [])
        alerts = st.session_state.get('alerts', [])
        
        # Calculate RMSE
        def calculate_rmse(y_true, y_pred):
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        adaline_rmse = calculate_rmse(y_val, adaline_pred)
        
        # Prepare data for visualization (inverse transform)
        actual_unscaled = scaler_y.inverse_transform(y_val)
        adaline_unscaled = scaler_y.inverse_transform(adaline_pred)
        
        # Main visualization section
        st.header("📊 Comprehensive Model Comparison")
        
        # Tab layout for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 3D Trajectories", "📈 Training Dynamics", "📉 Error Analysis", "🔬 Model Details"])
        
        with tab1:
            st.subheader("3D Trajectory Comparison")
            
            # Determine what to show
            show_lstm = 'lstm_pred' in st.session_state and 'lstm_pred' in st.session_state
            
            # 3D trajectory plot
            fig_3d = go.Figure()
            
            # Actual NASA path
            fig_3d.add_trace(go.Scatter3d(
                x=actual_unscaled[:, 0],
                y=actual_unscaled[:, 1],
                z=actual_unscaled[:, 2],
                mode='markers',
                name='Actual NASA Path',
                marker=dict(
                    size=3,
                    color=np.arange(len(actual_unscaled)),
                    colorscale='Viridis',
                    opacity=0.7,
                    showscale=True,
                    colorbar=dict(title="Time Step")
                )
            ))
            
            # Adaline prediction
            fig_3d.add_trace(go.Scatter3d(
                x=adaline_unscaled[:, 0],
                y=adaline_unscaled[:, 1],
                z=adaline_unscaled[:, 2],
                mode='lines',
                name='Linear Regression',
                line=dict(color='red', width=3)
            ))
            
            # LSTM prediction if available
            if show_lstm:
                lstm_pred = st.session_state.get('lstm_pred')
                y_val_seq = st.session_state.get('y_val_seq')
                if lstm_pred is not None and y_val_seq is not None:
                    # Align predictions
                    sequence_length = st.session_state.get('sequence_length', 10)
                    actual_vis = y_val[sequence_length-1:]
                    # Ensure same length
                    min_len = min(len(actual_vis), len(lstm_pred))
                    actual_vis = actual_vis[:min_len]
                    lstm_pred_aligned = lstm_pred[:min_len]
                    
                    lstm_unscaled = scaler_y.inverse_transform(lstm_pred_aligned)
                    actual_vis_unscaled = scaler_y.inverse_transform(actual_vis)
                
                fig_3d.add_trace(go.Scatter3d(
                    x=lstm_unscaled[:, 0],
                    y=lstm_unscaled[:, 1],
                    z=lstm_unscaled[:, 2],
                    mode='lines',
                    name='LSTM (Sequential Memory)',
                    line=dict(color='green', width=3)
                ))
            
            fig_3d.update_layout(
                title="3D Trajectory Comparison: Apophis Asteroid",
                scene=dict(
                    xaxis_title="X (AU)",
                    yaxis_title="Y (AU)",
                    zaxis_title="Z (AU)",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=600,
                legend=dict(x=0, y=1)
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Metrics comparison
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Linear Regression RMSE", f"{adaline_rmse:.6f}", "Validation")
                if show_lstm and 'lstm_pred' in st.session_state and 'y_val_seq' in st.session_state:
                    lstm_pred = st.session_state['lstm_pred']
                    y_val_seq = st.session_state['y_val_seq']
                    
                    # Calculate RMSE on the sequences (what LSTM was trained on)
                    lstm_rmse_seq = calculate_rmse(y_val_seq, lstm_pred)
                    
                    # Also calculate on aligned validation set (same as Adaline) for fair comparison
                    sequence_length = st.session_state.get('sequence_length', 10)
                    actual_vis = y_val[sequence_length-1:]
                    min_len = min(len(actual_vis), len(lstm_pred))
                    lstm_rmse_aligned = calculate_rmse(actual_vis[:min_len], lstm_pred[:min_len])
                    
                    with col2:
                        # Show both for transparency
                        st.metric("LSTM RMSE", f"{lstm_rmse_seq:.6f}", "On Sequences")
                        st.caption(f"Aligned: {lstm_rmse_aligned:.6f}")
                    with col3:
                        # Compare with Adaline using aligned RMSE
                        improvement = ((adaline_rmse - lstm_rmse_aligned) / adaline_rmse * 100)
                        st.metric("LSTM vs Linear Regression", f"{improvement:.2f}%", 
                                 delta="Better" if improvement > 0 else "Worse")
        
        with tab2:
            st.subheader("Training Dynamics: Error Surface Journey")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # MSE history
                fig_mse = go.Figure()
                fig_mse.add_trace(go.Scatter(
                    y=mse_history,
                    mode='lines+markers',
                    name='Linear Regression MSE',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ))
                
                if show_lstm and 'lstm_losses' in st.session_state:
                    lstm_losses = st.session_state['lstm_losses']
                    fig_mse.add_trace(go.Scatter(
                        y=lstm_losses,
                        mode='lines+markers',
                        name='LSTM Loss',
                        line=dict(color='green', width=2),
                        marker=dict(size=4)
                    ))
                
                # Highlight concept drift alerts
                if len(alerts) > 0:
                    for alert in alerts:
                        fig_mse.add_vline(
                            x=alert['epoch'],
                            line_dash="dot",
                            line_color="orange",
                            annotation_text="⚠️"
                        )
                
                fig_mse.update_layout(
                    title="Error Surface: The 'Bowl' Journey",
                    xaxis_title="Epoch",
                    yaxis_title="Loss (MSE)",
                    height=400,
                    hovermode='x unified',
                    legend=dict(x=0, y=1)
                )
                
                st.plotly_chart(fig_mse, use_container_width=True)
            
            with col2:
                # Physics constraint if enabled
                if use_pigd and len(physics_history) > 0:
                    fig_physics = go.Figure()
                    fig_physics.add_trace(go.Scatter(
                        y=physics_history,
                        mode='lines',
                        name='Physics Constraint',
                        line=dict(color='red', width=2)
                    ))
                    fig_physics.update_layout(
                        title="Physics Constraint (Kepler's Law)",
                        xaxis_title="Epoch",
                        yaxis_title="Constraint Violation",
                        height=400
                    )
                    st.plotly_chart(fig_physics, use_container_width=True)
                else:
                    st.info("Enable PIGD to see physics constraint visualization")
            
            # Learning rate analysis
            if learning_rate > 0.1:
                st.warning("⚠️ High learning rate detected! The 'ball' may bounce out of the 'bowl'.")
            elif learning_rate < 0.005:
                st.info("ℹ️ Low learning rate: Training will be slow but stable.")
            else:
                st.success("✅ Learning rate is in a good range for stable training.")
        
        with tab3:
            st.subheader("Error Analysis: Linear Regression vs LSTM")
            
            # Error over time
            adaline_error = np.linalg.norm(actual_unscaled - adaline_unscaled, axis=1)
            
            fig_error_time = go.Figure()
            fig_error_time.add_trace(go.Scatter(
                y=adaline_error,
                mode='lines',
                name='Linear Regression Error',
                line=dict(color='red', width=2)
            ))
            
            if show_lstm and 'lstm_pred' in st.session_state and 'y_val_seq' in st.session_state:
                lstm_pred = st.session_state['lstm_pred']
                y_val_seq = st.session_state['y_val_seq']
                sequence_length = st.session_state.get('sequence_length', 10)
                # Align actual data with LSTM predictions (account for sequence length)
                actual_vis = y_val[sequence_length-1:]
                # Ensure same length
                min_len = min(len(actual_vis), len(lstm_pred))
                actual_vis = actual_vis[:min_len]
                lstm_pred_aligned = lstm_pred[:min_len]
                
                lstm_unscaled = scaler_y.inverse_transform(lstm_pred_aligned)
                actual_vis_unscaled = scaler_y.inverse_transform(actual_vis)
                lstm_error = np.linalg.norm(actual_vis_unscaled - lstm_unscaled, axis=1)
                
                fig_error_time.add_trace(go.Scatter(
                    y=lstm_error,
                    mode='lines',
                    name='LSTM Error',
                    line=dict(color='green', width=2)
                ))
            
            fig_error_time.update_layout(
                title="Prediction Error Over Time",
                xaxis_title="Time Step",
                yaxis_title="Position Error (AU)",
                height=400,
                legend=dict(x=0, y=1)
            )
            
            st.plotly_chart(fig_error_time, use_container_width=True)
            
            # Error distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=adaline_error,
                    name='Linear Regression',
                    nbinsx=50,
                    opacity=0.7,
                    marker_color='red'
                ))
                
                if show_lstm:
                    fig_dist.add_trace(go.Histogram(
                        x=lstm_error,
                        name='LSTM',
                        nbinsx=50,
                        opacity=0.7,
                        marker_color='green'
                    ))
                
                fig_dist.update_layout(
                    title="Error Distribution",
                    xaxis_title="Position Error (AU)",
                    yaxis_title="Frequency",
                    height=400,
                    barmode='overlay',
                    legend=dict(x=0, y=1)
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Statistics
                st.subheader("Error Statistics")
                
                stats_data = {
                    'Model': ['Linear Regression'],
                    'Mean Error': [np.mean(adaline_error)],
                    'Std Error': [np.std(adaline_error)],
                    'Max Error': [np.max(adaline_error)],
                    'Min Error': [np.min(adaline_error)]
                }
                
                if show_lstm and 'lstm_pred' in st.session_state and 'y_val_seq' in st.session_state:
                    lstm_pred = st.session_state['lstm_pred']
                    y_val_seq = st.session_state['y_val_seq']
                    sequence_length = 10
                    actual_vis = y_val[sequence_length-1:]
                    # Ensure same length
                    min_len = min(len(actual_vis), len(lstm_pred))
                    actual_vis = actual_vis[:min_len]
                    lstm_pred_aligned = lstm_pred[:min_len]
                    
                    lstm_unscaled = scaler_y.inverse_transform(lstm_pred_aligned)
                    actual_vis_unscaled = scaler_y.inverse_transform(actual_vis)
                    lstm_error = np.linalg.norm(actual_vis_unscaled - lstm_unscaled, axis=1)
                    stats_data['Model'].append('LSTM')
                    stats_data['Mean Error'].append(np.mean(lstm_error))
                    stats_data['Std Error'].append(np.std(lstm_error))
                    stats_data['Max Error'].append(np.max(lstm_error))
                    stats_data['Min Error'].append(np.min(lstm_error))
                
                import pandas as pd
                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats, use_container_width=True, hide_index=True)
        
        with tab4:
            st.subheader("Model Details & Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Linear Regression Model")
                st.write(f"**Final MSE**: {mse_history[-1]:.6f}")
                st.write(f"**Initial MSE**: {mse_history[0]:.6f}")
                improvement = ((mse_history[0] - mse_history[-1]) / mse_history[0] * 100)
                st.write(f"**Improvement**: {improvement:.2f}%")
                st.write(f"**RMSE**: {adaline_rmse:.6f}")
                
                if use_pigd:
                    st.write(f"**Physics Constraint**: {physics_history[-1]:.6f}")
                
                if quantization_bits:
                    st.write(f"**Quantization**: {quantization_bits}-bit")
                
                st.write(f"**Concept Drift Alerts**: {len(alerts)}")
            
            with col2:
                if show_lstm and 'lstm_losses' in st.session_state and 'lstm_pred' in st.session_state:
                    st.markdown("### 🧠 LSTM Model")
                    lstm_losses = st.session_state['lstm_losses']
                    lstm_pred = st.session_state['lstm_pred']
                    y_val_seq = st.session_state.get('y_val_seq')
                    st.write(f"**Final Loss**: {lstm_losses[-1]:.6f}")
                    st.write(f"**Initial Loss**: {lstm_losses[0]:.6f}")
                    lstm_improvement = ((lstm_losses[0] - lstm_losses[-1]) / lstm_losses[0] * 100)
                    st.write(f"**Improvement**: {lstm_improvement:.2f}%")
                    if y_val_seq is not None:
                        # CRITICAL: LSTM was trained to predict y_val_seq
                        # The correct RMSE is comparing lstm_pred directly to y_val_seq
                        if len(y_val_seq) != len(lstm_pred):
                            st.error(f"❌ Shape mismatch: y_val_seq {y_val_seq.shape} vs lstm_pred {lstm_pred.shape}")
                            st.error("This indicates a problem with LSTM training or prediction!")
                            lstm_rmse = 0.0
                        else:
                            # This is the CORRECT RMSE - LSTM predicts y_val_seq, so compare to y_val_seq
                            lstm_rmse = calculate_rmse(y_val_seq, lstm_pred)
                            st.write(f"**RMSE (Validation)**: {lstm_rmse:.6f}")
                            
                            # Verify alignment: y_val_seq should align with y_val[sequence_length-1:]
                            sequence_length = st.session_state.get('sequence_length', 10)
                            actual_vis = y_val[sequence_length-1:]
                            
                            if len(actual_vis) == len(y_val_seq):
                                # They should be the same - verify
                                lstm_rmse_check = calculate_rmse(actual_vis, lstm_pred)
                                if abs(lstm_rmse - lstm_rmse_check) > 0.001:
                                    st.warning(f"⚠️ Alignment check: {lstm_rmse_check:.6f} (should match)")
                        
                        # Overfitting indicator
                        if lstm_losses[-1] < 0.001 and lstm_rmse > 0.3:
                            st.warning("⚠️ **Overfitting Detected**: Low training loss but high validation RMSE. Try:")
                            st.write("  - Reducing LSTM epochs")
                            st.write("  - Lowering learning rate")
                            st.write("  - Adding dropout regularization")
                    st.write(f"**Architecture**: 2 LSTM layers (128 hidden dims, dropout=0.0)")
                    if quantization_bits:
                        st.write(f"**Quantization**: {quantization_bits}-bit")
            
            # Concept drift alerts
            if len(alerts) > 0:
                st.markdown("### ⚠️ Concept Drift Alerts")
                for alert in alerts:
                    st.warning(f"**Epoch {alert['epoch']}**: {alert['message']} (Error increase: {alert['error_increase']:.6f})")
            
            # Error surface visualization
            st.markdown("### 🎯 Error Surface Visualization")
            st.markdown("""
            The error surface is like a "bowl" - the bottom represents optimal weights.
            As learning rate increases, the optimization "ball" may bounce out of the bowl.
            """)
            
            # Simplified error surface
            w_range = np.linspace(-2, 2, 50)
            error_surface = np.zeros((50, 50))
            for i, w1 in enumerate(w_range):
                for j, w2 in enumerate(w_range):
                    error_surface[i, j] = w1**2 + w2**2
            
            fig_surface = go.Figure(data=[go.Surface(z=error_surface, x=w_range, y=w_range, colorscale='Viridis')])
            
            if len(mse_history) > 10:
                trajectory_z = mse_history[::max(1, len(mse_history)//20)]
                trajectory_x = np.linspace(-1, 1, len(trajectory_z))
                trajectory_y = np.linspace(-1, 1, len(trajectory_z))
                
                fig_surface.add_trace(go.Scatter3d(
                    x=trajectory_x,
                    y=trajectory_y,
                    z=trajectory_z,
                    mode='markers+lines',
                    name='Training Path',
                    marker=dict(size=5, color='red'),
                    line=dict(color='red', width=3)
                ))
            
            fig_surface.update_layout(
                title="Error Surface: The 'Bowl' (Simplified 2D Projection)",
                scene=dict(
                    xaxis_title="Weight 1",
                    yaxis_title="Weight 2",
                    zaxis_title="Error (MSE)",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                ),
                height=500
            )
            
            st.plotly_chart(fig_surface, use_container_width=True)
            
            # Educational content
            with st.expander("📚 Understanding the Models"):
                st.markdown("""
                **Linear Regression:**
                - Uses Widrow-Hoff Delta Rule: w = w + η(y - ŷ)x
                - Linear activation function for regression
                - Struggles with non-linear trajectories
                
                **LSTM:**
                - Maintains hidden state memory across time steps
                - Captures sequential patterns and non-linear dynamics
                - Better at predicting curved orbital trajectories
                
                **Physics-Informed Gradient Descent (PIGD):**
                - Adds Kepler's Second Law constraint (angular momentum conservation)
                - Demonstrates hybrid AI: data-driven + physics knowledge
                - Helps model respect physical laws while learning
                
                **Concept Drift Detection:**
                - Monitors error trends over time
                - Alerts when linear models can't track curved trajectories
                - Simulates historical moment when MLPs were needed
                """)
    
    else:
        st.info("👈 Adjust parameters in the sidebar and click 'Train Models' to start!")
        
        # Show data info
        st.subheader("📊 Data Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Validation Samples", len(X_val))
        with col3:
            st.metric("Features", f"{X_train.shape[1]} (Time, X, Y, Z, VX, VY, VZ)")

else:
    st.error("Failed to load data. Please check your internet connection and try again.")
