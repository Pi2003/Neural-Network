import numpy as np
import pandas as pd
import copy

# Helper functions
def he_initialization(shape):
    """
    Initialize weights using He normal initialization.

    Parameters:
        shape (tuple): Shape of the weight matrix (fan_in, fan_out).

    Returns:
        np.ndarray: Initialized weight matrix.
    """
    fan_in = shape[0] # Number of input neurons
    std_dev = np.sqrt(2/fan_in) # He normal initialization formula 
    return np.array(np.random.randn(*shape) * std_dev, dtype=np.float64)

def xavier_initialization(shape):
    """
    Initialize weights using Xavier normal initialization.

    Parameters:
        shape (tuple): Shape of the weight matrix (fan_in, fan_out).

    Returns:
        np.ndarray: Initialized weight matrix.
    """
    fan_in, fan_out = shape # Number of input and output neurons 
    std_dev = np.sqrt(2 / (fan_in + fan_out)) # Xavier normal initialization formula
    return np.array(np.random.randn(*shape) * std_dev, dtype=np.float64)

# Initialize set of weights
def initialize_biases (shape):
    """
    Initialize biases as zeros.

    Parameters:
        shape (tuple): Shape of the bias vector.

    Returns:
        np.ndarray: Zero-initialized bias vector.
    """ 
    return np.zeros(shape)

def initialize_biases_random(shape):
    """
    Initialize biases randomly with small values.

    Parameters:
        shape (tuple): Shape of the bias vector.

    Returns:
        np.ndarray: Randomly initialized bias vector.
    """
    return np.random.randn(*shape) * 0.01

# Activation functions
def relu(x):
    """
    Apply the ReLU activation function.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Output after applying ReLU.
    """
    return np.maximum(0, x)

def leaky_relu(x):
    """
    Apply the Leaky ReLU activation function.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Output after applying Leaky ReLU.
    """
    return np.maximum (0.01 * x, x)

def sigmoid(x):
    """
    Apply the sigmoid activation function.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Output after applying sigmoid.
    """
    x = np.clip(x, -500, 1000)
    return 1/(1 + np.exp(-x))

def tanh(x):
    """
    Apply the hyperbolic tangent activation function.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Output after applying tanh.
    """
    return np.tanh(x)

def softmax(matrix):
    """
    Apply the softmax function to each row of the input matrix.

    Parameters:
        matrix (np.ndarray): Input 2D array.

    Returns:
        np.ndarray: Softmax probabilities for each row.
    """
    exp_matrix = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
    #Compute the softmax for each row
    softmax_matrix = exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)
    return softmax_matrix

def calculate_cross_entropy(actual, predicted):
    """
    Compute the cross-entropy loss for classification.

    Parameters:
        actual (np.ndarray): One-hot encoded true labels.
        predicted (np.ndarray): Predicted probabilities.

    Returns:
        float: Cross-entropy loss.
    """
    # For classification
    # Ensure the inputs are numpy arrays
    actual = np.array(actual)
    predicted =  np.array(predicted)

    # Clip predicted values to avoid log(0)
    predicted = np.clip(predicted, 1e-12, 1.0)

    #Calculate cross-entropy loss
    cross_entropy = np.mean(-np.sum(actual * np.log(predicted), axis = 1))
    return cross_entropy

def calculate_mse(actual, predicted):
    """
    Compute the mean squared error (MSE) loss for regression.

    Parameters:
        actual (np.ndarray): True target values.
        predicted (np.ndarray): Predicted values.

    Returns:
        float: Mean squared error.
    """
    # For regression
    # Ensure the inputs are numpy arrays
    actual = np.array(actual)
    predicted =  np.array(predicted)

    # Clip predicted values to avoid log(0)
    predicted = np.clip(predicted, 1e-12, 1.0)

    #Calculate cross-entropy loss
    cross_entropy = np.mean(-np.sum(actual * np.log(predicted), axis = 1))
    return cross_entropy

def relu_derivative(x):
    """
    Compute the derivative of the ReLU activation function.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Derivative of ReLU.
    """
    return np.where(x > 0, 1, 0)

def leaky_relu_derivative(x):
    """
    Compute the derivative of the Leaky ReLU activation function.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Derivative of Leaky ReLU.
    """
    return np.where(x > 0, 1, 0.01)

def sigmoid_derivative(x):
    """
    Compute the derivative of the sigmoid activation function.

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Derivative of sigmoid.
    """
    sig = sigmoid(x)
    return sig * (1-sig)

# Neural network functions
def initialization(features_count, neuron_count):
    """
    Initialize weights and biases for each layer of the network.

    Parameters:
        neuron_count (list): List of neuron counts for each layer.

    Returns:
        tuple: Tuple containing lists of weight matrices and bias vectors.
    """
    print("Initialization...")
    input_features = features_count
    weights_matrix = []
    bias_matrix =[]

    for count in neuron_count:
        weights_matrix.append(he_initialization((input_features, count)))
        bias_matrix.append(initialize_biases_random((count, 1)))
        input_features = count
    
    # Convert lists to numpy arrays
    weights_matrix = np.array(weights_matrix, dtype = object)
    bias_matrix = np.array(bias_matrix, dtype = object)
    return weights_matrix, bias_matrix

def classification_prediction(X, classification_model, hidden_layers):
    """
    Make predictions for classification tasks using the trained model.

    Parameters:
        X (pd.DataFrame): Input features.
        classification_model (tuple): Tuple of weights and biases.
        hidden_layers (int): Number of hidden layers.

    Returns:
        np.ndarray: Predicted class indices.
    """
    print("Predicting...")
    X_np = X.to_numpy()
    result = X_np.copy()
    
    # Hidden layer calculations
    for i in range(hidden_layers):
        #Layer output = activation_function(Weights * Input + Bias)
        result = leaky_relu(np.dot(result, classification_model[0][i]) + classification_model[1][i].transpose())
    
    # Output layer calculations
    result = softmax(np.dot(result, classification_model[0][-1]) + classification_model[1][-1].transpose())
    return np.argmax(result, axis=1)

def forward_propagation_classification(X, y_one_hot_encoded, hidden_layers, weights_matrix, bias_matrix):
    """
    Perform forward propagation for a classification neural network.

    Parameters:
        X (pd.DataFrame): Input features.
        y_one_hot_encoded (np.ndarray): One-hot encoded true labels.
        hidden_layers (int): Number of hidden layers.
        weights_matrix (list): List of weight matrices.
        bias_matrix (list): List of bias vectors.

    Returns:
        tuple: (List of layer outputs, cross-entropy loss)
    """
    print("Forward-propagation...")
    X_np = X.to_numpy()
    result = X_np.copy()
    layer_outputs = []
    layer_outputs.append(result)
    # Hidden layer calculations
    for i in range(hidden_layers):
        #Layer output = activation_function(Weights * Input + Bias)
        result = leaky_relu(np.dot(result, weights_matrix[i]) + bias_matrix[i].transpose())
        layer_outputs.append(result)

    # Output layer calculations
    result = softmax(np.dot(result, weights_matrix[-1]) + bias_matrix[-1].transpose())
    layer_outputs.append(result)
    return layer_outputs, calculate_cross_entropy(y_one_hot_encoded, result)

def backward_propagation_classification(y_one_hot_encoded, layer_outputs, hidden_layers, weights_matrix, bias_matrix, learning_rate):
    """
    Perform backward propagation and update weights and biases for classification.

    Parameters:
        y_one_hot_encoded (np.ndarray): One-hot encoded true labels.
        layer_outputs (list): Outputs from each layer during forward pass.
        hidden_layers (int): Number of hidden layers.
        weights_matrix (list): List of weight matrices.
        bias_matrix (list): List of bias vectors.
        learning_rate (float): Learning rate for updates.

    Returns:
        tuple: Updated weights and biases.
    """
    print("Back-propagation...")
    # Final layer
    loss_wrt_final_layer = layer_outputs[-1] - y_one_hot_encoded
    loss_wrt_weights_final_layer = np.dot(layer_outputs[-2].transpose(), loss_wrt_final_layer)
    loss_wrt_bias_final_layer = loss_wrt_final_layer.transpose().mean(axis=1).reshape(bias_matrix[-1].shape[0], 1)

    loss_wrt_layers = []
    loss_wrt_weights_layers = []
    loss_wrt_bias_layers = []

    loss_wrt_layers.append(loss_wrt_final_layer)
    loss_wrt_weights_layers.append(loss_wrt_weights_final_layer)
    loss_wrt_bias_layers.append(loss_wrt_bias_final_layer)

    for i in range(1, hidden_layers + 1):
        loss_wrt_layers.insert(0, np.dot(loss_wrt_layers[0],
                                         weights_matrix[-i].transpose()) * leaky_relu_derivative(layer_outputs[-i-1]))
        loss_wrt_weights_layers.insert(0, np.dot(layer_outputs[-i-2].transpose(), 
                                                 loss_wrt_layers[0]))
        loss_wrt_bias_layers.insert(0, loss_wrt_layers[0].transpose().mean(axis = 1).reshape(bias_matrix[-i-1].shape[0], 1))

    # Weights and bias readjusment
    for i in  range(hidden_layers + 1):
        weights_matrix[i] -= learning_rate * loss_wrt_weights_layers[i]
        bias_matrix[i] -= learning_rate * loss_wrt_bias_layers[i]

    return weights_matrix, bias_matrix

def classification_network_training( X, y_one_hot_encoded, 
                                    neuron_count, epochs, learning_rate,
                                    hidden_layers):
    """
    Train a neural network for classification tasks.

    Parameters:
        X (pd.DataFrame): Input features.
        y_one_hot_encoded (np.ndarray): One-hot encoded true labels.
        neuron_count (list): List of neuron counts for each layer.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate.
        hidden_layers (int): Number of hidden layers.

    Returns:
        tuple: Best model (weights, biases) found during training.
    """
    # Initialization
    weights_matrix, bias_matrix = initialization(len(X.columns), neuron_count)
    # Forward propagation
    layer_outputs, loss = forward_propagation_classification(X, 
                                                            y_one_hot_encoded, 
                                                            hidden_layers,
                                                            weights_matrix,
                                                            bias_matrix)
    
    best_loss = loss
    best_model = (copy.deepcopy(weights_matrix), copy.deepcopy(bias_matrix))

    for epoch in range(epochs):
        # Backpropagation
        weights_matrix, bias_matrix = backward_propagation_classification(y_one_hot_encoded,
                                                                          layer_outputs,
                                                                          hidden_layers,
                                                                          weights_matrix,
                                                                          bias_matrix,
                                                                          learning_rate)
        
        # Forward propagation
        layer_outputs, loss = forward_propagation_classification(X,
                                                                y_one_hot_encoded,
                                                                hidden_layers,
                                                                weights_matrix,
                                                                bias_matrix)
        
        print(f"Loss for epoch {epoch + 1}: ", loss)
        if not np.isnan(loss) and loss < best_loss:
            # overwrite the model if loss improves
            print("Improved loss: ", loss)
            best_model = (copy.deepcopy(weights_matrix), copy.deepcopy(bias_matrix))
    return best_model

def regression_prediction(X, classification_model, hidden_layers):
    """
    Make predictions for regression tasks using the trained model.

    Parameters:
        X (pd.DataFrame): Input features.
        classification_model (tuple): Tuple of weights and biases.
        hidden_layers (int): Number of hidden layers.

    Returns:
        np.ndarray: Predicted values.
    """
    print("Predicting...")
    X_np = X.to_numpy()
    result = X_np.copy()
    
    # Hidden layer calculations
    for i in range(hidden_layers):
        #Layer output = activation_function(Weights * Input + Bias)
        result = leaky_relu(np.dot(result, classification_model[0][i]) + classification_model[1][i].transpose())
    
    # Output layer calculations
    result = np.dot(result, classification_model[0][-1]) + classification_model[1][-1].transpose()
    return result

def forward_propagation_regression(X, y, hidden_layers, weights_matrix, bias_matrix):
    """
    Perform forward propagation for a regression neural network.

    Parameters:
        X (pd.DataFrame): Input features.
        y (pd.Series or np.ndarray): True target values.
        hidden_layers (int): Number of hidden layers.
        weights_matrix (list): List of weight matrices.
        bias_matrix (list): List of bias vectors.

    Returns:
        tuple: (List of layer outputs, mean squared error loss)
    """
    print("Forward-propagation...")
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    result = X_np.copy()
    layer_outputs = []
    layer_outputs.append(result)
    # Hidden layer calculations
    for i in range(hidden_layers):
        #Layer output = activation_function(Weights * Input + Bias)
        result = leaky_relu(np.dot(result, weights_matrix[i]) + bias_matrix[i].transpose())
        layer_outputs.append(result)

    # Output layer calculations
    result = np.dot(result, weights_matrix[-1]) + bias_matrix[-1].transpose()
    layer_outputs.append(result)
    return layer_outputs, calculate_mse(y_np.reshape(result.shape), result)

def backward_propagation_regression(y, layer_outputs, hidden_layers, weights_matrix, bias_matrix, learning_rate):
    """
    Perform backward propagation and update weights and biases for regression.

    Parameters:
        y (pd.Series or np.ndarray): True target values.
        layer_outputs (list): Outputs from each layer during forward pass.
        hidden_layers (int): Number of hidden layers.
        weights_matrix (list): List of weight matrices.
        bias_matrix (list): List of bias vectors.
        learning_rate (float): Learning rate for updates.

    Returns:
        tuple: Updated weights and biases.
    """
    print("Back-propagation...")
    y_np = y.to_numpy()
    y_np = y_np.reshape(layer_outputs[-1].shape)
    # Final layer
    # Ignore the coefficient as that would be handled by the learning rate over epochs
    loss_wrt_final_layer = layer_outputs[-1] - y_np
    loss_wrt_weights_final_layer = np.dot(layer_outputs[-2].transpose(), loss_wrt_final_layer)
    loss_wrt_bias_final_layer = loss_wrt_final_layer.transpose().mean(axis=1).reshape(bias_matrix[-1].shape[0], 1)

    loss_wrt_layers = []
    loss_wrt_weights_layers = []
    loss_wrt_bias_layers = []

    loss_wrt_layers.append(loss_wrt_final_layer)
    loss_wrt_weights_layers.append(loss_wrt_weights_final_layer)
    loss_wrt_bias_layers.append(loss_wrt_bias_final_layer)

    for i in range(1, hidden_layers + 1):
        loss_wrt_layers.insert(0, np.dot(loss_wrt_layers[0],
                                         weights_matrix[-i].transpose()) * leaky_relu_derivative(layer_outputs[-i-1]))
        loss_wrt_weights_layers.insert(0, np.dot(layer_outputs[-i-2].transpose(), 
                                                 loss_wrt_layers[0]))
        loss_wrt_bias_layers.insert(0, loss_wrt_layers[0].transpose().mean(axis = 1).reshape(bias_matrix[-i-1].shape[0], 1))

    # Weights and bias readjusment
    for i in  range(hidden_layers + 1):
        weights_matrix[i] -= learning_rate * loss_wrt_weights_layers[i]
        bias_matrix[i] -= learning_rate * loss_wrt_bias_layers[i]

    return weights_matrix, bias_matrix

def regression_network_training( X, y, 
                                    neuron_count, epochs, learning_rate,
                                    hidden_layers):
    """
    Train a neural network for regression tasks.

    Parameters:
        X (pd.DataFrame): Input features.
        y (pd.Series or np.ndarray): True target values.
        neuron_count (list): List of neuron counts for each layer.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate.
        hidden_layers (int): Number of hidden layers.

    Returns:
        tuple: Best model (weights, biases) found during training.
    """
    # Initialization
    weights_matrix, bias_matrix = initialization(len(X.columns), neuron_count)
    # Forward propagation
    layer_outputs, loss = forward_propagation_regression(X, 
                                                        y, 
                                                        hidden_layers,
                                                        weights_matrix,
                                                        bias_matrix)
    
    best_loss = loss
    best_model = (copy.deepcopy(weights_matrix), copy.deepcopy(bias_matrix))

    for epoch in range(epochs):
        # Backpropagation
        weights_matrix, bias_matrix = backward_propagation_regression(y,
                                                                    layer_outputs,
                                                                    hidden_layers,
                                                                    weights_matrix,
                                                                    bias_matrix,
                                                                    learning_rate)
        
        # Forward propagation
        layer_outputs, loss = forward_propagation_regression(X,
                                                            y,
                                                            hidden_layers,
                                                            weights_matrix,
                                                            bias_matrix)
        
        print(f"Loss for epoch {epoch + 1}: ", loss)
        if not np.isnan(loss) and loss < best_loss:
            # overwrite the model if loss improves
            print("Improved loss: ", loss)
            best_model = (copy.deepcopy(weights_matrix), copy.deepcopy(bias_matrix))
    return best_model