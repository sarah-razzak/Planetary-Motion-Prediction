"""
Data Acquisition Module for The Ancient Predictor

This module fetches telemetry data for Asteroid Apophis (99942) from NASA JPL Horizons
and prepares it for training historical (Adaline) and modern (LSTM) models.

The data represents the "Geometry of Data" - the fundamental challenge that Gauss
faced when trying to predict planetary motion using Least Squares, and that Rosenblatt
attempted to solve with the Perceptron and Adaline architectures.
"""

import numpy as np
import pandas as pd
from astroquery.jplhorizons import Horizons
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta


def fetch_apophis_data(start_date='2026-01-01', end_date='2030-01-01'):
    """
    Fetch daily telemetry data for Asteroid Apophis (99942) from NASA JPL Horizons.
    
    The normalization using StandardScaler is critical: without it, gradient descent
    oscillates wildly, unable to find the "bottom of the bowl" in the error surface.
    This is the same challenge that plagued early optimization attempts in the 1960s.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    X : np.ndarray
        Features: [Time, X, Y, Z, VX, VY, VZ] for each day
    y : np.ndarray
        Labels: [X, Y, Z] position at t+1
    scaler_X : StandardScaler
        Fitted scaler for features (for inverse transform if needed)
    scaler_y : StandardScaler
        Fitted scaler for labels (for inverse transform if needed)
    dates : np.ndarray
        Array of datetime objects for each data point
    """
    print("Fetching Apophis (99942) data from NASA JPL Horizons...")
    print(f"Date range: {start_date} to {end_date}")
    
    # Create date range
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    dates = []
    current = start
    
    # Fetch data day by day
    data_points = []
    
    while current <= end:
        dates.append(current)
        date_str = current.strftime('%Y-%m-%d')
        
        try:
            obj = Horizons(id='99942', location='@sun', epochs={'start': date_str, 'stop': date_str, 'step': '1d'})
            eph = obj.ephemerides()
            
            if len(eph) > 0:
                # Extract position and velocity
                x = eph['x'][0]  # AU
                y = eph['y'][0]  # AU
                z = eph['z'][0]  # AU
                vx = eph['vx'][0]  # AU/day
                vy = eph['vy'][0]  # AU/day
                vz = eph['vz'][0]  # AU/day
                
                # Time feature: days since start
                t = (current - start).days
                
                data_points.append([t, x, y, z, vx, vy, vz])
            else:
                print(f"Warning: No data for {date_str}")
                
        except Exception as e:
            print(f"Error fetching data for {date_str}: {e}")
        
        current += timedelta(days=1)
    
    if len(data_points) < 2:
        raise ValueError("Insufficient data fetched. Please check date range and internet connection.")
    
    data_array = np.array(data_points)
    dates_array = np.array(dates)
    
    # Prepare features (X) and labels (y)
    # Features: [Time, X, Y, Z, VX, VY, VZ] at time t
    # Labels: [X, Y, Z] at time t+1
    X = data_array[:-1]  # All but last row
    y = data_array[1:, 1:4]  # X, Y, Z of next time step
    
    # Normalize features and labels
    # This is crucial: without normalization, the gradient descent algorithm
    # struggles with the different scales of time, position (AU), and velocity (AU/day).
    # StandardScaler ensures all features contribute equally to the error surface.
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    print(f"Successfully fetched {len(X_scaled)} data points")
    print(f"Feature shape: {X_scaled.shape}")
    print(f"Label shape: {y_scaled.shape}")
    
    return X_scaled, y_scaled, scaler_X, scaler_y, dates_array[:-1]


def prepare_sequences(X, y, sequence_length=10):
    """
    Prepare sequences for LSTM training.
    
    The LSTM requires sequences because it maintains "hidden states" - a form of memory
    that allows it to capture the non-linear dynamics of elliptical orbits that simple
    linear models like Adaline cannot represent.
    
    Parameters:
    -----------
    X : np.ndarray
        Scaled features
    y : np.ndarray
        Scaled labels
    sequence_length : int
        Length of input sequences for LSTM
    
    Returns:
    --------
    X_seq : np.ndarray
        Sequences of shape (n_sequences, sequence_length, n_features)
    y_seq : np.ndarray
        Targets of shape (n_sequences, n_outputs)
    """
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length-1])
    
    return np.array(X_seq), np.array(y_seq)

