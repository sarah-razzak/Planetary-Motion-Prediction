# The Ancient Predictor: From Gauss to LSTM

> *"The Geometry of Data meets the Gradient Descent"*

A journey through machine learning history, exploring how 18th-century Least Squares (Gauss) evolved into 1960s Adaline (Widrow-Hoff) and ultimately to 2026-era LSTMs, all applied to the timeless problem of predicting planetary motion.

## 📚 Historical Context

This project is inspired by **Anil Ananthaswamy's "Why Machines Learn"** (Chapters 1-5), which explores:

- **The Geometry of Data**: How mathematical models represent relationships in high-dimensional space
- **Gradient Descent**: The optimization technique that finds the "bottom of the bowl" in error surfaces
- **The Geometrical Crisis**: Why linear models struggle with non-linear phenomena like elliptical orbits

### The Evolution

1. **18th Century - Gauss's Least Squares**: The foundation of regression, used to predict planetary positions from noisy observations
2. **1960s - Adaline (Widrow-Hoff)**: An adaptive linear neuron that learns through the Delta Rule, bridging Gauss and Rosenblatt
3. **2026-era - LSTM**: Long Short-Term Memory networks that capture sequential patterns through hidden states

## 🎯 Project Overview

**The Ancient Predictor** implements all three approaches to predict the trajectory of **Asteroid Apophis (99942)** from January 2026 to January 2030, using real telemetry data from NASA JPL Horizons.

### Key Features

- **Data Acquisition**: Automated fetching of asteroid telemetry from NASA JPL Horizons
- **Adaline Implementation**: Pure NumPy implementation of the Widrow-Hoff Delta Rule
- **LSTM Baseline**: PyTorch-based sequential model with hidden state memory
- **3D Visualization**: Comparative analysis of actual vs. predicted trajectories
- **Computational Efficiency**: Benchmarking inference times for onboard satellite applications

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd planetary_motion_with_LMS
```

2. Install dependencies:
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
planetary_motion_with_LMS/
├── data.py          # Data acquisition and preprocessing
├── model.py         # Adaline and LSTM model implementations
├── main.py          # Main execution script
├── requirements.txt # Python dependencies
└── README.md        # This file
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

## 🎓 Educational Value

This project serves as a bridge between:
- **Historical Mathematics**: Gauss's Least Squares and the geometry of optimization
- **1960s Machine Learning**: Rosenblatt's Perceptron and Widrow-Hoff's Adaline
- **Modern Deep Learning**: Recurrent neural networks and sequential modeling

The code comments extensively reference:
- The "Error Surface" and gradient descent
- The "Geometrical Crisis" of linear models
- How hidden states solve non-linearity challenges

## 📝 Commit Messages

Here are 5 high-quality commit messages that tell the story of the project:

1. **Initial commit: Data acquisition module with NASA JPL Horizons integration**
   - Implemented `fetch_apophis_data()` using astroquery.jplhorizons
   - Added StandardScaler normalization to prevent gradient descent oscillations
   - Created sequence preparation utilities for LSTM training

2. **Implement Adaline model with Widrow-Hoff Delta Rule**
   - Pure NumPy implementation of 1960s Adaline architecture
   - Linear activation function for regression in 3D space
   - MSE tracking to visualize error surface journey

3. **Add PyTorch LSTM model with hidden state memory**
   - Implemented LSTMPredictor class with 64-dim hidden states
   - Sequence-based training to capture orbital dynamics
   - Comments explaining how hidden states solve non-linearity

4. **Create 3D visualization and benchmarking pipeline**
   - Matplotlib 3D plots comparing actual vs. predicted trajectories
   - RMSE calculation and error distribution analysis
   - Computational efficiency measurement for satellite applications

5. **Complete documentation and project structure**
   - Comprehensive README bridging Gauss, Rosenblatt, and modern ML
   - Requirements.txt with all dependencies
   - Educational code comments referencing "Geometrical Crisis"

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

