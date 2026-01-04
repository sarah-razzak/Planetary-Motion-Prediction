"""
Interactive Error Surface Dashboard

A Streamlit dashboard for visualizing the error surface and training dynamics
of the Adaline model. Demonstrates how learning rate affects gradient descent
and the "ball in bowl" analogy.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px

from data import fetch_apophis_data, prepare_sequences
from model import Adaline
import torch
import torch.nn as nn

# Page config
st.set_page_config(
    page_title="The Ancient Predictor - Error Surface Dashboard",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 The Ancient Predictor: Error Surface Dashboard")
st.markdown("### Interactive visualization of gradient descent and the error surface")

# Sidebar controls
st.sidebar.header("Training Parameters")

learning_rate = st.sidebar.slider(
    "Learning Rate (η)",
    min_value=0.001,
    max_value=0.5,
    value=0.01,
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
    help="Simulate edge chip constraints (2026 feature)"
)

epochs = st.sidebar.slider(
    "Training Epochs",
    min_value=10,
    max_value=200,
    value=100,
    step=10
)

# Load data
@st.cache_data
def load_data():
    """Load and cache the asteroid data."""
    try:
        X, y, scaler_X, scaler_y, dates = fetch_apophis_data('2026-01-01', '2026-12-31')
        # Use smaller subset for dashboard
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        return X_train, y_train, X_val, y_val, scaler_X, scaler_y
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None

X_train, y_train, X_val, y_val, scaler_X, scaler_y = load_data()

if X_train is not None:
    # Train model
    if st.sidebar.button("🚀 Train Model", type="primary"):
        with st.spinner("Training Adaline model..."):
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
            st.session_state['mse_history'] = adaline.mse_history
            st.session_state['physics_history'] = adaline.physics_constraint_history
            st.session_state['alerts'] = adaline.concept_drift_alerts
    
    # Display results if model is trained
    if 'adaline' in st.session_state:
        adaline = st.session_state['adaline']
        mse_history = st.session_state['mse_history']
        physics_history = st.session_state['physics_history']
        alerts = st.session_state['alerts']
        
        # Create two columns for visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📉 Error Surface Journey (MSE)")
            
            # Error surface visualization
            fig = go.Figure()
            
            # Plot MSE history
            fig.add_trace(go.Scatter(
                y=mse_history,
                mode='lines+markers',
                name='MSE',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            # Add physics constraint if enabled
            if use_pigd and len(physics_history) > 0:
                fig.add_trace(go.Scatter(
                    y=np.array(physics_history) * lambda_pigd,
                    mode='lines',
                    name='Physics Constraint',
                    line=dict(color='red', width=2, dash='dash'),
                    yaxis='y2'
                ))
            
            # Highlight concept drift alerts
            if len(alerts) > 0:
                for alert in alerts:
                    fig.add_vline(
                        x=alert['epoch'],
                        line_dash="dot",
                        line_color="orange",
                        annotation_text="⚠️ Alert"
                    )
            
            fig.update_layout(
                title="Error Surface: The 'Bowl' Journey",
                xaxis_title="Epoch",
                yaxis_title="MSE",
                yaxis2=dict(
                    title="Physics Constraint",
                    overlaying='y',
                    side='right'
                ) if use_pigd else None,
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Learning rate warning
            if learning_rate > 0.1:
                st.warning("⚠️ High learning rate detected! The 'ball' may bounce out of the 'bowl'.")
            elif learning_rate < 0.005:
                st.info("ℹ️ Low learning rate: Training will be slow but stable.")
        
        with col2:
            st.subheader("🌍 3D Trajectory Prediction")
            
            # Make predictions
            y_pred = adaline.predict(X_val)
            
            # Inverse transform to original scale
            actual_unscaled = scaler_y.inverse_transform(y_val[:100])  # First 100 for visualization
            pred_unscaled = scaler_y.inverse_transform(y_pred[:100])
            
            # 3D plot
            fig_3d = go.Figure()
            
            # Actual trajectory
            fig_3d.add_trace(go.Scatter3d(
                x=actual_unscaled[:, 0],
                y=actual_unscaled[:, 1],
                z=actual_unscaled[:, 2],
                mode='markers',
                name='Actual NASA Path',
                marker=dict(size=2, color='blue', opacity=0.6)
            ))
            
            # Predicted trajectory
            fig_3d.add_trace(go.Scatter3d(
                x=pred_unscaled[:, 0],
                y=pred_unscaled[:, 1],
                z=pred_unscaled[:, 2],
                mode='lines',
                name='Adaline Prediction',
                line=dict(color='red', width=3)
            ))
            
            fig_3d.update_layout(
                title="3D Trajectory Comparison",
                scene=dict(
                    xaxis_title="X (AU)",
                    yaxis_title="Y (AU)",
                    zaxis_title="Z (AU)"
                ),
                height=400
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
        
        # Metrics row
        col3, col4, col5 = st.columns(3)
        
        with col3:
            final_mse = mse_history[-1] if len(mse_history) > 0 else 0
            initial_mse = mse_history[0] if len(mse_history) > 0 else 0
            improvement = ((initial_mse - final_mse) / initial_mse * 100) if initial_mse > 0 else 0
            
            st.metric(
                "Final MSE",
                f"{final_mse:.6f}",
                delta=f"{improvement:.1f}% improvement"
            )
        
        with col4:
            if use_pigd:
                final_physics = physics_history[-1] if len(physics_history) > 0 else 0
                st.metric(
                    "Physics Constraint",
                    f"{final_physics:.6f}",
                    delta="Lower is better"
                )
            else:
                st.metric("Quantization", "Full Precision" if quantization_bits is None else f"{quantization_bits}-bit")
        
        with col5:
            st.metric(
                "Concept Drift Alerts",
                len(alerts),
                delta="Non-linearity detected" if len(alerts) > 0 else "No alerts"
            )
        
        # Concept drift alerts
        if len(alerts) > 0:
            st.subheader("⚠️ Concept Drift Alerts")
            for alert in alerts:
                st.warning(f"**Epoch {alert['epoch']}**: {alert['message']} (Error increase: {alert['error_increase']:.6f})")
        
        # Error surface 2D visualization
        st.subheader("🎯 Error Surface Visualization")
        st.markdown("""
        This visualization shows the "bowl" shape of the error surface. 
        As the learning rate increases, the optimization "ball" may bounce out of the bowl.
        """)
        
        # Create a simplified 2D error surface
        w_range = np.linspace(-2, 2, 50)
        error_surface = np.zeros((50, 50))
        
        # Simplified error surface (bowl shape)
        for i, w1 in enumerate(w_range):
            for j, w2 in enumerate(w_range):
                # Parabolic bowl: error increases with distance from center
                error_surface[i, j] = w1**2 + w2**2
        
        fig_surface = go.Figure(data=[go.Surface(z=error_surface, x=w_range, y=w_range)])
        
        # Add trajectory point (simplified)
        if len(mse_history) > 10:
            # Show how MSE decreases (moving toward center of bowl)
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
                zaxis_title="Error (MSE)"
            ),
            height=500
        )
        
        st.plotly_chart(fig_surface, use_container_width=True)
        
        # Explanation
        with st.expander("📚 Understanding the Error Surface"):
            st.markdown("""
            **The Error Surface ("Bowl") Analogy:**
            
            1. **The Bowl**: The error surface is like a bowl - the bottom represents the optimal weights
            2. **The Ball**: Your model's weights are like a ball rolling down the bowl
            3. **Learning Rate (η)**: Controls how big the steps are
               - Too small: Ball moves slowly, takes forever to reach bottom
               - Too large: Ball bounces out of the bowl (divergence)
               - Just right: Ball smoothly rolls to the bottom
            
            **Physics-Informed Gradient Descent (PIGD):**
            - Adds a constraint based on Kepler's Second Law (angular momentum conservation)
            - Helps the model respect physical laws while learning from data
            - Demonstrates hybrid AI: combining data-driven learning with human knowledge
            
            **Concept Drift Detection:**
            - Monitors error trends over time
            - Alerts when linear models can't track curved trajectories
            - Simulates the historical moment when researchers realized they needed MLPs
            """)
    
    else:
        st.info("👈 Adjust parameters in the sidebar and click 'Train Model' to start!")
        
        # Show data info
        st.subheader("📊 Data Information")
        st.write(f"Training samples: {len(X_train)}")
        st.write(f"Validation samples: {len(X_val)}")
        st.write(f"Features: {X_train.shape[1]} (Time, X, Y, Z, VX, VY, VZ)")
        st.write(f"Labels: {y_train.shape[1]} (X, Y, Z position at t+1)")

else:
    st.error("Failed to load data. Please check your internet connection and try again.")

