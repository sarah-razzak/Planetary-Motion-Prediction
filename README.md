# The Ancient Predictor: From Gauss to LSTM

> *"The Geometry of Data meets the Gradient Descent"*

A machine learning project that implements Adaline (1960s) and LSTM (2026-era) models to predict the trajectory of **Asteroid Apophis (99942)** using real telemetry data from NASA JPL Horizons.

## 🎯 Project Overview

**The Ancient Predictor** compares historical and modern machine learning approaches to predict planetary motion, using real telemetry data from NASA JPL Horizons.

### Key Features

- **Data Acquisition**: Automated fetching of asteroid telemetry from NASA JPL Horizons
- **Adaline Implementation**: Pure NumPy implementation of the Widrow-Hoff Delta Rule
- **LSTM Baseline**: PyTorch-based sequential model with hidden state memory
- **3D Visualization**: Comparative analysis of actual vs. predicted trajectories
- **Computational Efficiency**: Benchmarking inference times for onboard satellite applications

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher (Python 3.12 recommended)
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone https://github.com/sarah-razzak/Planetary-Motion-Prediction.git
cd Planetary-Motion-Prediction
```

2. Create a virtual environment:
```bash
python3 -m venv venv
```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Running the Complete Pipeline

Execute the main script to run the entire pipeline:

```bash
python main.py
```

This will:
1. Fetch Apophis data from NASA JPL Horizons (Jan 2026 - Jan 2030)
2. Normalize the data using StandardScaler
3. Train the Adaline model (100 epochs)
4. Train the LSTM model (50 epochs)
5. Generate 3D visualizations comparing predictions
6. Print performance metrics and computational efficiency

### Module Structure

```
Planetary-Motion-Prediction/
├── data.py          # Data acquisition and preprocessing
├── model.py         # Adaline and LSTM model implementations
├── main.py          # Main execution script
├── requirements.txt # Python dependencies
├── README.md        # This file
└── venv/            # Virtual environment (not in git)
```

## 🔬 Technical Details

### Data Format

The model uses 7 input features:
- **Time** (t): Days since start date
- **Position** (X, Y, Z): Cartesian coordinates in Astronomical Units (AU)
- **Velocity** (VX, VY, VZ): Velocity components in AU/day

**Label**: Position (X, Y, Z) at time t+1

### Adaline Model

The Adaline (Adaptive Linear Neuron) uses the **Widrow-Hoff Delta Rule**:

\[
w = w + \eta(y - \hat{y})x
\]

where:
- \(w\): weight vector
- \(\eta\) (eta): learning rate
- \(y\): true label
- \(\hat{y}\): predicted output (\(\mathbf{w}^T \mathbf{x} + b\))
- \(x\): input feature vector

**Key Characteristics**:
- Linear activation function (identity) for regression
- Batch gradient descent
- Tracks MSE history to visualize error surface journey

### LSTM Model

The LSTM architecture:
- **Input**: Sequences of 10 time steps
- **LSTM Layer**: 64 hidden dimensions, 1 layer
- **Output Head**: Linear layer mapping hidden state to 3D position

**Key Innovation**: Hidden states maintain memory of previous positions and velocities, enabling the model to capture non-linear orbital dynamics that linear models cannot represent.

## 📊 Results Interpretation

### The Geometrical Crisis

The visualization demonstrates a fundamental limitation of linear models:

- **Adaline** approximates the trajectory with straight-line segments, struggling with the curved nature of elliptical orbits
- **LSTM** captures the orbital dynamics through sequential memory, producing smoother, more accurate predictions

### Performance Metrics

The script outputs:
- **RMSE** (Root Mean Square Error) for both models
- **Inference Time** (milliseconds) for computational efficiency analysis
- **3D Trajectory Plots** comparing actual NASA data with predictions
- **Error Distribution** histograms

## 🔗 References

- Ananthaswamy, A. (2024). *Why Machines Learn: The Elegant Math Behind Modern AI*. Dutton.
- Widrow, B., & Hoff, M. E. (1960). "Adaptive switching circuits." IRE WESCON Convention Record.
- Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." Neural computation.

## 📄 License

This project is for educational purposes, demonstrating the historical evolution of machine learning through the lens of planetary motion prediction.

## 🙏 Acknowledgments

- NASA JPL Horizons for providing asteroid ephemeris data
- The mathematical foundations laid by Gauss, Rosenblatt, and Widrow-Hoff
- The deep learning community for advancing sequential modeling

---

*"In the geometry of data, we find the patterns that govern motion—from the orbits of planets to the descent of gradients."*
