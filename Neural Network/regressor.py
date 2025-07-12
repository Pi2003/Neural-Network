#%%
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

#%%
dataset = pd.read_csv(r"D:/Neural Network/data/BostonHousing.csv")
df = dataset.loc[:, :]
df = df.dropna()

#%% Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop("medv", axis=1))
x_cols = list(df.columns)
x_cols.remove("medv")

#%%
X = pd.DataFrame(scaled_data, columns = x_cols)
y = df["medv"]

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% User input
# Python lists (not numpy arrays)
target_column = "medv"
reg_class = int(input('Choose if regression or classification:\nRegression => 0\nClassification => 1: ')) # 1

output_activation = 'softmax' if reg_class == 1 else 'Linear'
hidden_layers = int(input('Enter number of hidden Layers: ')) # 2

neuron_count = []
for i in range(hidden_layers):
    neuron_count.append(int(input(f'Enter number of neurons in layer {i + 1}:'))) # 30
neuron_count.append(1) # If regression model

learning_rate = float(input('Enter the Learning rate for the network: '))
epochs = int(input('Enter the number of epochs for the network: '))

# %%
#%% Helper functions

def he_initialization(shape):
    """He using NumPy"""
    fan_in = shape[0] # Number of input neurons
    std_dev = np.sqrt(2/fan_in) # He normal initialization formula 
    return np.array(np.random.randn(*shape) * std_dev, dtype=np.float64)

def xavier_initialization(shape):
    """Xavier (Glorot) initialization using NumPy"""
    fan_in, fan_out = shape # Number of input and output neurons 
    std_dev = np.sqrt(2 / (fan_in + fan_out)) # Xavier normal initialization formula
    return np.array(np.random.randn(*shape) * std_dev, dtype=np.float64)

# Initialize set of weights
def initialize_biases (shape): 
    return np.zeros(shape)

def initialize_biases_random(shape):
    return np.random.randn(*shape) * 0.01

# Activation functions
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.maximum (0.01 * x, x)

def sigmoid(x):
    x = np.clip(x, -1000, 1000)
    return 1/(1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(matrix):
    exp_matrix = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
    #Compute the softmax for each row
    softmax_matrix = exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)
    return softmax_matrix

def calculate_mse(actual, predicted):
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
    return np.where(x > 0, 1, 0)

def leaky_relu_derivative(x):
    return np.where(x > 0, 1, 0.01)

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1-sig)

#%% Neural network functions
def initialization(neuron_count):
    print("Initialization...")
    input_features = len(X.columns)
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

def regression_prediction(X, classification_model, hidden_layers):
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

#%%
def regression_network_training( X, y, 
                                    neuron_count, epochs, learning_rate,
                                    hidden_layers):
    # Initialization
    weights_matrix, bias_matrix = initialization(neuron_count)
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

#%%
regression_model = regression_network_training(X_train, y_train, neuron_count, epochs, learning_rate, hidden_layers)

#%%
predictions = regression_prediction(X_test, regression_model, hidden_layers)
actual_values = y_test.to_numpy().reshape(y_test.shape[0], 1)

#%%
mse = mean_squared_error(actual_values, predictions)
mae = mean_absolute_error(actual_values, predictions)
r2 = r2_score(actual_values, predictions)
rmse = np.sqrt(mse)


print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")
print(f"Root Mean Squared Error (MSE): {rmse}")