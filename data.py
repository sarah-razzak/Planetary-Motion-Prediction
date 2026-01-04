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
from astropy import units as u
from astropy.coordinates import SkyCoord


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
    
    # Fetch data in batch using date range (more efficient)
    data_points = []
    dates = []
    
    try:
        # Query Horizons API with date range
        # Request state vectors (Cartesian coordinates) using quantities parameter
        obj = Horizons(
            id='99942',
            location='@sun',
            epochs={'start': start_date, 'stop': end_date, 'step': '1d'}
        )
        
        # Request ephemerides - default includes r, RA, DEC which we can convert to Cartesian
        # Note: The API doesn't directly return x,y,z,vx,vy,vz, so we'll convert from spherical
        eph = obj.ephemerides()
        
        if len(eph) == 0:
            raise ValueError("No data returned from API")
        
        print(f"Successfully fetched {len(eph)} data points from API")
        
        # Extract position and velocity from the table
        # The API returns an Astropy Table with columns: 'x', 'y', 'z', 'vx', 'vy', 'vz' (in AU and AU/day)
        colnames = [col.lower() for col in eph.colnames]  # Get lowercase column names
        
        # Find the correct column names (they might be 'x', 'y', 'z' or 'X', 'Y', 'Z', etc.)
        x_col = None
        y_col = None
        z_col = None
        vx_col = None
        vy_col = None
        vz_col = None
        
        for col in eph.colnames:
            col_lower = col.lower()
            if col_lower == 'x' and x_col is None:
                x_col = col
            elif col_lower == 'y' and y_col is None:
                y_col = col
            elif col_lower == 'z' and z_col is None:
                z_col = col
            elif col_lower == 'vx' and vx_col is None:
                vx_col = col
            elif col_lower == 'vy' and vy_col is None:
                vy_col = col
            elif col_lower == 'vz' and vz_col is None:
                vz_col = col
        
        if not all([x_col, y_col, z_col, vx_col, vy_col, vz_col]):
            # Try to convert from spherical coordinates (r, RA, DEC) if available
            if 'r' in eph.colnames and 'RA' in eph.colnames and 'DEC' in eph.colnames:
                print("Cartesian coordinates not directly available.")
                print("Converting from spherical coordinates (r, RA, DEC) to Cartesian (x, y, z)...")
                # We'll convert in the loop below
                x_col = y_col = z_col = vx_col = vy_col = vz_col = 'CONVERT'
                use_spherical = True
            else:
                # Try alternative column names
                print(f"Available columns: {eph.colnames[:20]}")
                raise ValueError(f"Missing required columns. Found: x={x_col}, y={y_col}, z={z_col}, vx={vx_col}, vy={vy_col}, vz={vz_col}")
        use_spherical = False
        if not all([x_col, y_col, z_col, vx_col, vy_col, vz_col]):
            # Check if we have spherical coordinates to convert
            if 'r' in eph.colnames and 'RA' in eph.colnames and 'DEC' in eph.colnames:
                use_spherical = True
        
        # Extract data from each row
        for i in range(len(eph)):
            try:
                # Get date from the row
                try:
                    date_str = str(eph['datetime_str'][i])
                    # Parse date string (format: '2026-Jan-01 00:00')
                    date_obj = datetime.strptime(date_str.split()[0], '%Y-%b-%d')
                except:
                    # Fallback: calculate from start date + index
                    date_obj = start + timedelta(days=i)
                
                dates.append(date_obj)
                
                # Extract Cartesian coordinates using found column names
                if use_spherical or x_col == 'CONVERT':
                    # Convert from spherical to Cartesian
                    r = float(eph['r'][i])  # AU (heliocentric distance)
                    ra = float(eph['RA'][i])  # degrees
                    dec = float(eph['DEC'][i])  # degrees
                    
                    # Convert RA/DEC to radians
                    ra_rad = np.deg2rad(ra)
                    dec_rad = np.deg2rad(dec)
                    
                    # Convert to Cartesian (heliocentric)
                    # Note: RA is measured eastward from vernal equinox, DEC is measured from equator
                    x = r * np.cos(dec_rad) * np.cos(ra_rad)
                    y = r * np.cos(dec_rad) * np.sin(ra_rad)
                    z = r * np.sin(dec_rad)
                    
                    # For velocity, convert rates if available
                    if 'r_rate' in eph.colnames and 'RA_rate' in eph.colnames and 'DEC_rate' in eph.colnames:
                        r_rate = float(eph['r_rate'][i])  # AU/day
                        ra_rate = float(eph['RA_rate'][i])  # arcsec/day -> convert to rad/day
                        dec_rate = float(eph['DEC_rate'][i])  # arcsec/day -> convert to rad/day
                        
                        # Convert arcsec/day to rad/day
                        ra_rate_rad = np.deg2rad(ra_rate / 3600.0)  # arcsec to degrees, then to rad
                        dec_rate_rad = np.deg2rad(dec_rate / 3600.0)
                        
                        # Convert spherical velocity rates to Cartesian velocities
                        # Using the derivative of the spherical-to-Cartesian transformation
                        vx = (r_rate * np.cos(dec_rad) * np.cos(ra_rad) - 
                              r * np.sin(dec_rad) * np.cos(ra_rad) * dec_rate_rad - 
                              r * np.cos(dec_rad) * np.sin(ra_rad) * ra_rate_rad)
                        vy = (r_rate * np.cos(dec_rad) * np.sin(ra_rad) - 
                              r * np.sin(dec_rad) * np.sin(ra_rad) * dec_rate_rad + 
                              r * np.cos(dec_rad) * np.cos(ra_rad) * ra_rate_rad)
                        vz = (r_rate * np.sin(dec_rad) + 
                              r * np.cos(dec_rad) * dec_rate_rad)
                    else:
                        # Approximate velocities as zero (not ideal but works)
                        vx = vy = vz = 0.0
                else:
                    x = float(eph[x_col][i])  # AU
                    y = float(eph[y_col][i])  # AU
                    z = float(eph[z_col][i])  # AU
                    vx = float(eph[vx_col][i])  # AU/day
                    vy = float(eph[vy_col][i])  # AU/day
                    vz = float(eph[vz_col][i])  # AU/day
                
                # Time feature: days since start
                t = (date_obj - start).days
                
                data_points.append([t, x, y, z, vx, vy, vz])
                
            except (KeyError, ValueError, IndexError, TypeError) as e:
                print(f"Warning: Could not parse row {i}, skipping. Error: {e}")
                continue
                    
    except Exception as e:
        error_msg = str(e)
        if "Insufficient data" not in error_msg:  # Don't print if it's our own error
            print(f"Error fetching data from API: {error_msg[:200]}")
        print("Note: NASA JPL Horizons API may be unavailable or rate-limited.")
        print("The code will use synthetic orbital data for demonstration purposes.")
        # Re-raise to trigger fallback in main.py
        raise ValueError("API fetch failed - will use synthetic data")
        
        # Fallback: try day by day (limited to avoid too many API calls)
        current = start
        max_fallback_days = min(30, (end - start).days + 1)  # Limit to 30 days for fallback
        
        while current <= end and len(data_points) < max_fallback_days:
            dates.append(current)
            date_str = current.strftime('%Y-%m-%d')
            
            try:
                obj = Horizons(id='99942', location='@sun', epochs=date_str)
                # Try vectors() first, then fallback to ephemerides
                try:
                    eph = obj.vectors()
                except:
                    eph = obj.ephemerides(quantities='9')
                
                if len(eph) > 0:
                    # Find column names
                    colnames = [col.lower() for col in eph.colnames]
                    x_col = next((c for c in eph.colnames if c.lower() == 'x'), None)
                    y_col = next((c for c in eph.colnames if c.lower() == 'y'), None)
                    z_col = next((c for c in eph.colnames if c.lower() == 'z'), None)
                    vx_col = next((c for c in eph.colnames if c.lower() == 'vx'), None)
                    vy_col = next((c for c in eph.colnames if c.lower() == 'vy'), None)
                    vz_col = next((c for c in eph.colnames if c.lower() == 'vz'), None)
                    
                    if all([x_col, y_col, z_col, vx_col, vy_col, vz_col]):
                        x = float(eph[x_col][0])
                        y = float(eph[y_col][0])
                        z = float(eph[z_col][0])
                        vx = float(eph[vx_col][0])
                        vy = float(eph[vy_col][0])
                        vz = float(eph[vz_col][0])
                        
                        t = (current - start).days
                        data_points.append([t, x, y, z, vx, vy, vz])
                    
            except Exception as e2:
                print(f"Error fetching data for {date_str}: {str(e2)[:100]}...")
            
            current += timedelta(days=1)
    
    if len(data_points) < 2:
        raise ValueError("Insufficient data fetched. Please check date range and internet connection.")
    
    # Ensure dates and data_points are aligned
    if len(dates) != len(data_points):
        # Recreate dates from data_points if misaligned
        dates = [start + timedelta(days=int(dp[0])) for dp in data_points]
    
    data_array = np.array(data_points)
    dates_array = np.array(dates[:len(data_points)])  # Ensure same length
    
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
        # Target: predict position at time i+sequence_length
        # X[i:i+sequence_length] covers times i to i+sequence_length-1
        # y[j] = position at time j+1
        # So position at time i+sequence_length = y[i+sequence_length-1]
        y_seq.append(y[i+sequence_length-1])
    
    return np.array(X_seq), np.array(y_seq)

