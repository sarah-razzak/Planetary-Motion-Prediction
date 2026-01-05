# Linear Regression vs LSTM: Planetary Motion Prediction

> *"The Geometry of Data meets the Gradient Descent"*

A machine learning project comparing **Linear Regression** (1960s) and **LSTM** (2026-era) models to predict the trajectory of **Asteroid Apophis (99942)** using real telemetry data from NASA JPL Horizons.

🌐 **[Try the Interactive Dashboard →](https://planetary-motion-prediction-acaxpjgcydubjxbnaoyvwd.streamlit.app)**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

## 🎯 Project Overview

This project compares **Linear Regression** and **LSTM** models to predict planetary motion, using real telemetry data from NASA JPL Horizons. The comparison demonstrates how linear models struggle with non-linear orbital dynamics, while recurrent architectures excel at capturing sequential patterns.

### Key Features

- **Data Acquisition**: Automated fetching of asteroid telemetry from NASA JPL Horizons
- **Linear Regression**: Pure NumPy implementation using the Widrow-Hoff Delta Rule
- **LSTM Model**: PyTorch-based sequential model with hidden state memory
- **3D Visualization**: Comparative analysis of actual vs. predicted trajectories
- **Interactive Dashboard**: [Live Streamlit app](https://planetary-motion-prediction-acaxpjgcydubjxbnaoyvwd.streamlit.app) for real-time experimentation
- **Computational Efficiency**: Benchmarking inference times for onboard satellite applications

### 2026 Enhancements

- **Physics-Informed Gradient Descent (PIGD)**: Adds Kepler's Second Law constraint to loss function
- **Concept Drift Detection**: Monitors error trends to detect when linear models fail
- **Quantization Support**: Simulates edge chip constraints (4-bit, 8-bit) for both models
- **Interactive Dashboard**: [Live Streamlit app](https://planetary-motion-prediction-acaxpjgcydubjxbnaoyvwd.streamlit.app) for error surface visualization with learning rate controls

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher (Python 3.12 recommended)
- pip package manager

### Setup

1. Clone this repository:
```bash
git clone https://github.com/sarah-razzak/planetary_motion_with_LMS.git
cd planetary_motion_with_LMS
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

### Quick Start

The fastest way to see the project in action:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python main.py
```

### Running the Complete Pipeline

Execute the main script to run the entire pipeline:

```bash
python main.py
```

This will:
1. Fetch Apophis data from NASA JPL Horizons (Jan 2026 - Jan 2030)
2. Normalize the data using StandardScaler
3. Train the Linear Regression model (150 epochs) with optional PIGD
4. Train the LSTM model (150 epochs with early stopping)
5. Test quantization robustness (4-bit, 8-bit)
6. Generate 3D visualizations comparing predictions
7. Print performance metrics and computational efficiency

### Interactive Dashboard

🌐 **[Access the Live Dashboard](https://planetary-motion-prediction-acaxpjgcydubjxbnaoyvwd.streamlit.app)**

Or run locally:

```bash
streamlit run dashboard.py
```

The dashboard allows you to:
- Adjust learning rate (η) and see its effect on training
- Enable/disable Physics-Informed Gradient Descent (PIGD)
- Test quantization levels (4-bit, 8-bit)
- Visualize the error surface "bowl" in real-time
- See concept drift alerts when linear models struggle
- Compare Linear Regression vs LSTM performance interactively

### Project Structure

```
planetary_motion_with_LMS/
├── data.py              # Data acquisition and preprocessing
├── model.py             # Linear Regression and LSTM model implementations
├── main.py              # Main execution script
├── dashboard.py         # Interactive Streamlit dashboard
├── optimize.py          # Hyperparameter optimization utilities
├── quick_optimize.py    # Quick optimization script
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── LICENSE              # MIT License
└── .gitignore           # Git ignore rules
```

## 🔬 Technical Details

### Data Format

The model uses 7 input features:
- **Time** (t): Days since start date
- **Position** (X, Y, Z): Cartesian coordinates in Astronomical Units (AU)
- **Velocity** (VX, VY, VZ): Velocity components in AU/day

**Label**: Position (X, Y, Z) at time t+1

### Linear Regression Model

The Linear Regression model uses the **Widrow-Hoff Delta Rule** for weight updates:

\[
\mathbf{w} = \mathbf{w} + \eta(y - \hat{y})\mathbf{x}
\]

where:
- \(\mathbf{w}\): weight vector
- \(\eta\) (eta): learning rate
- \(y\): true label (position at t+1)
- \(\hat{y}\): predicted output
- \(\mathbf{x}\): input feature vector

The prediction is computed as:

\[
\hat{y} = \mathbf{w}^T \mathbf{x} + b
\]

The loss function is Mean Squared Error (MSE):

\[
\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\]

**Key Characteristics**:
- Linear activation function (identity) for regression
- Batch gradient descent
- Tracks MSE history to visualize error surface journey
- Assumes linear relationship: \(y = f(\mathbf{x}) = \mathbf{w}^T\mathbf{x} + b\)

**Physics-Informed Variant (PIGD)**:

When enabled, the loss function incorporates Kepler's Second Law (angular momentum conservation):

\[
\mathcal{L}_{PIGD} = \mathcal{L}_{MSE} + \lambda \cdot \mathcal{L}_{physics}
\]

where \(\mathcal{L}_{physics} = \|\mathbf{L}_{predicted} - \mathbf{L}_{current}\|^2\) and \(\mathbf{L} = \mathbf{r} \times \mathbf{v}\) is the angular momentum vector.

### LSTM Model

The LSTM (Long Short-Term Memory) architecture uses gating mechanisms to maintain long-term dependencies:

**LSTM Cell Equations**:

\[
\begin{align}
f_t &= \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(forget gate)} \\
i_t &= \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(input gate)} \\
\tilde{C}_t &= \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) \quad \text{(candidate values)} \\
C_t &= f_t * C_{t-1} + i_t * \tilde{C}_t \quad \text{(cell state)} \\
o_t &= \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(output gate)} \\
h_t &= o_t * \tanh(C_t) \quad \text{(hidden state)}
\end{align}
\]

where:
- \(h_t\): hidden state at time \(t\)
- \(C_t\): cell state at time \(t\)
- \(x_t\): input at time \(t\)
- \(\sigma\): sigmoid activation function
- \(W_f, W_i, W_C, W_o\): weight matrices
- \(b_f, b_i, b_C, b_o\): bias vectors

**Architecture**:
- **Input**: Sequences of 10 time steps
- **LSTM Layer**: 128 hidden dimensions, 2 layers
- **Output Head**: Linear layer mapping hidden state to 3D position

**Prediction**:

\[
\hat{y}_t = W_{out} \cdot h_t + b_{out}
\]

**Key Innovation**: Hidden states maintain memory of previous positions and velocities, enabling the model to capture non-linear orbital dynamics that linear models cannot represent. The LSTM can learn functions of the form \(y_t = f(x_t, x_{t-1}, ..., x_{t-n})\), where \(f\) is non-linear.

## 📊 Results Interpretation

### The Geometrical Crisis

The visualization demonstrates a fundamental limitation of linear models:

- **Linear Regression** approximates the trajectory with straight-line segments, struggling with the curved nature of elliptical orbits. It assumes \(y = \mathbf{w}^T\mathbf{x} + b\), which cannot capture the non-linear relationship \(y = f(\mathbf{x}, \mathbf{x}_{t-1}, ...)\) required for orbital dynamics.
- **LSTM** captures the orbital dynamics through sequential memory, producing smoother, more accurate predictions. The hidden state \(h_t\) encodes information from all previous time steps, allowing it to learn non-linear patterns.

**Mathematical Intuition**: Planetary orbits follow elliptical paths described by:

\[
r = \frac{a(1-e^2)}{1+e\cos(\theta)}
\]

where \(r\) is the distance, \(a\) is the semi-major axis, \(e\) is eccentricity, and \(\theta\) is the true anomaly. Linear models approximate this as \(r \approx w_1\theta + w_2\), while LSTM learns the true non-linear relationship.

### Performance Metrics

The script outputs:
- **RMSE** (Root Mean Square Error) for both models
- **Inference Time** (milliseconds) for computational efficiency analysis
- **3D Trajectory Plots** comparing actual NASA data with predictions
- **Error Distribution** histograms

### Example Output

After running `python main.py`, you'll see:
- Training progress for both Linear Regression and LSTM models
- Quantization robustness analysis (4-bit, 8-bit, full precision)
- A saved visualization (`trajectory_comparison.png`) showing the 3D trajectory comparison
- Performance summary comparing model accuracy and inference speed

Or explore interactively in the [live dashboard](https://planetary-motion-prediction-acaxpjgcydubjxbnaoyvwd.streamlit.app)!

## 🔗 References

- Ananthaswamy, A. (2024). *Why Machines Learn: The Elegant Math Behind Modern AI*. Dutton.
- Widrow, B., & Hoff, M. E. (1960). "Adaptive switching circuits." IRE WESCON Convention Record.
- Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." Neural computation.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project is for educational purposes, demonstrating the historical evolution of machine learning through the lens of planetary motion prediction.

## 🙏 Acknowledgments

- NASA JPL Horizons for providing asteroid ephemeris data
- The mathematical foundations laid by Gauss (Least Squares), Rosenblatt (Perceptron), and Widrow-Hoff (Delta Rule)
- The deep learning community for advancing sequential modeling with LSTM architectures

---

*"In the geometry of data, we find the patterns that govern motion—from the orbits of planets to the descent of gradients."*

## 📝 Notes

- The project uses real NASA JPL Horizons data when available, with automatic fallback to synthetic orbital data if the API is unavailable
- All visualizations are saved automatically for later analysis
- The interactive dashboard requires an active internet connection for the initial data fetch
