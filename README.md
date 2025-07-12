# Neural Network from Scratch: Pure Mathematics Implementation

## 🧠 Overview

This project implements a **complete neural network from scratch** using only fundamental mathematical operations and NumPy. No high-level machine learning frameworks like TensorFlow or PyTorch are used—just pure mathematics, linear algebra, and calculus brought to life in code.

The implementation demonstrates the core mathematical principles behind deep learning, making it an excellent educational resource for understanding how neural networks actually work under the hood.

## 🎯 Key Features

### Mathematical Foundation
- **Pure Implementation**: Built entirely from mathematical first principles
- **No ML Frameworks**: Uses only NumPy for numerical computations
- **Educational Focus**: Every function is thoroughly documented with mathematical explanations

### Supported Tasks
- **Classification**: Multi-class classification with softmax output
- **Regression**: Continuous value prediction with linear output
- **Flexible Architecture**: Configurable hidden layers and neuron counts

### Advanced Features
- **Multiple Weight Initialization Schemes**: He Normal and Xavier Normal
- **Diverse Activation Functions**: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax
- **Robust Loss Functions**: Cross-entropy for classification, MSE for regression
- **Gradient Descent Optimization**: Full backpropagation implementation
- **Model Persistence**: Best model tracking during training

## 🔬 Mathematical Concepts Implemented

### 1. Weight Initialization
**He Normal Initialization** (for ReLU-based networks):
```
W ~ N(0, √(2/fan_in))
```

**Xavier Normal Initialization** (for symmetric activations):
```
W ~ N(0, √(2/(fan_in + fan_out)))
```

### 2. Forward Propagation
For each layer `l`:
```
Z^(l) = W^(l) · A^(l-1) + b^(l)
A^(l) = activation(Z^(l))
```

### 3. Activation Functions
- **ReLU**: `f(x) = max(0, x)`
- **Leaky ReLU**: `f(x) = max(0.01x, x)`
- **Sigmoid**: `f(x) = 1/(1 + e^(-x))`
- **Softmax**: `f(x_i) = e^(x_i) / Σ(e^(x_j))`

### 4. Loss Functions
**Cross-Entropy** (Classification):
```
L = -1/m Σ Σ y_ij · log(ŷ_ij)
```

**Mean Squared Error** (Regression):
```
L = 1/m Σ (y_i - ŷ_i)²
```

### 5. Backpropagation
Gradient computation using chain rule:
```
∂L/∂W^(l) = ∂L/∂A^(l) · ∂A^(l)/∂Z^(l) · ∂Z^(l)/∂W^(l)
```

## 📁 Project Structure

```
neural_network/
├── paste.txt                 # Main implementation file
├── README.md                 # This file
└── examples/                 # Usage examples (if added)
```

## 🚀 Quick Start

### Installation
```bash
pip install numpy pandas
```

### Basic Usage

#### Classification Example
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load your data
X = pd.DataFrame(your_features)
y = pd.Series(your_labels)

# One-hot encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_one_hot = np.eye(len(np.unique(y_encoded)))[y_encoded]

# Define network architecture
neuron_count = [64, 32, len(np.unique(y_encoded))]  # 2 hidden layers + output
hidden_layers = 2
epochs = 1000
learning_rate = 0.01

# Train the network
model = classification_network_training(
    X, y_one_hot, neuron_count, epochs, learning_rate, hidden_layers
)

# Make predictions
predictions = classification_prediction(X_test, model, hidden_layers)
```

#### Regression Example
```python
# Load your data
X = pd.DataFrame(your_features)
y = pd.Series(your_targets)

# Define network architecture
neuron_count = [64, 32, 1]  # 2 hidden layers + 1 output neuron
hidden_layers = 2
epochs = 1000
learning_rate = 0.001

# Train the network
model = regression_network_training(
    X, y, neuron_count, epochs, learning_rate, hidden_layers
)

# Make predictions
predictions = regression_prediction(X_test, model, hidden_layers)
```

## 🔧 API Reference

### Core Functions

#### Weight Initialization
- `he_initialization(shape)`: He Normal initialization for ReLU networks
- `xavier_initialization(shape)`: Xavier Normal initialization for symmetric activations
- `initialize_biases(shape)`: Zero initialization for biases
- `initialize_biases_random(shape)`: Small random initialization for biases

#### Activation Functions
- `relu(x)`: Rectified Linear Unit
- `leaky_relu(x)`: Leaky ReLU with α=0.01
- `sigmoid(x)`: Sigmoid activation
- `tanh(x)`: Hyperbolic tangent
- `softmax(matrix)`: Softmax for multi-class classification

#### Loss Functions
- `calculate_cross_entropy(actual, predicted)`: Cross-entropy loss
- `calculate_mse(actual, predicted)`: Mean squared error loss

#### Training Functions
- `classification_network_training(X, y_one_hot, neuron_count, epochs, learning_rate, hidden_layers)`: Train classification network
- `regression_network_training(X, y, neuron_count, epochs, learning_rate, hidden_layers)`: Train regression network

#### Prediction Functions
- `classification_prediction(X, model, hidden_layers)`: Make classification predictions
- `regression_prediction(X, model, hidden_layers)`: Make regression predictions

## ⚙️ Configuration Options

### Network Architecture
- **neuron_count**: List defining neurons per layer `[hidden1, hidden2, ..., output]`
- **hidden_layers**: Number of hidden layers
- **epochs**: Training iterations
- **learning_rate**: Step size for gradient descent

### Activation Functions
The implementation uses **Leaky ReLU** for hidden layers by default, which helps prevent the "dying ReLU" problem while maintaining computational efficiency.

### Weight Initialization
- **He Normal**: Default for ReLU-based networks
- **Xavier Normal**: Available for symmetric activations
- **Random Bias**: Small random values to break symmetry

## 📊 Performance Considerations

### Numerical Stability
- **Gradient Clipping**: Prevents exploding gradients
- **Epsilon Clipping**: Prevents log(0) in loss calculations
- **Overflow Protection**: Sigmoid function clipped to prevent overflow

### Memory Efficiency
- **In-place Operations**: Minimizes memory allocation
- **Efficient Matrix Operations**: Leverages NumPy's optimized BLAS
- **Best Model Tracking**: Saves only the best performing model

## 🎓 Educational Value

This implementation is designed to be educational. Each function includes:
- **Mathematical formulation** in docstrings
- **Step-by-step computation** with clear variable names
- **Conceptual explanations** of why each operation is performed
- **Numerical stability considerations**

Perfect for:
- Students learning neural networks
- Researchers implementing custom architectures
- Anyone wanting to understand the mathematical foundations of deep learning

## 🐛 Known Limitations

1. **No GPU Support**: CPU-only implementation
2. **Basic Optimization**: Only vanilla gradient descent
3. **Limited Regularization**: No dropout, batch normalization, or weight decay
4. **Single Batch**: Processes entire dataset at once (no mini-batching)
5. **Fixed Architecture**: No skip connections or advanced architectures

## 🔮 Future Enhancements

- [ ] Mini-batch gradient descent
- [ ] Advanced optimizers (Adam, RMSprop)
- [ ] Regularization techniques
- [ ] Batch normalization
- [ ] Convolutional layers
- [ ] GPU acceleration with CuPy
- [ ] Model serialization/deserialization
- [ ] Advanced metrics and visualization

## 📚 Mathematical References

- **Deep Learning** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- **Neural Networks and Deep Learning** by Michael Nielsen
- **Pattern Recognition and Machine Learning** by Christopher Bishop

## 🤝 Contributing

This is an educational implementation. Contributions that improve:
- Mathematical clarity
- Code documentation
- Educational examples
- Performance optimizations

are welcome!

## 📄 License

MIT License - Feel free to use this for educational purposes, research, or as a foundation for your own implementations.

---

**Built with ❤️ and Pure Mathematics**

*"The best way to understand neural networks is to build them from scratch."*