#%%
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

#%%
dataset = pd.read_csv(r"D:/Neural Network/data/iris_dataset.csv")
df = dataset.loc[:, :]
df = df.dropna()

#%% Encode categorical columns
class_encodings = {'setosa': 0, 'virginica': 1, 'versicolor': 2}

df['target'] = df['target'].str.strip()
df['target'] = df['target'].map(class_encodings).astype(int)

X = df.drop('target', axis = 1)
count_labels = len(df['target'].unique())
y = df['target']

#%% Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_train = np.eye(count_labels)[y_train]
y_test = np.eye(count_labels)[y_test]

#%% User input
# Python lists (not numpy arrays)
target_column = 'target'
reg_class = int(input('Choose if regression or classification:\nRegression => 0\nClassification => 1: ')) # 1

output_activation = 'softmax' if reg_class == 1 else 'Linear'
hidden_layers = int(input('Enter number of hidden Layers: ')) # 2

neuron_count = []
for i in range(hidden_layers):
    neuron_count.append(int(input(f'Enter number of neurons in layer {i + 1}:'))) # 30

neuron_count.append(count_labels) # If classfication model

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
    x = np.clip(x, -500, 1000)
    return 1/(1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(matrix):
    exp_matrix = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
    #Compute the softmax for each row
    softmax_matrix = exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)
    return softmax_matrix

def calculate_cross_entropy(actual, predicted):
    # For classification
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

def classification_prediction(X, classification_model, hidden_layers):
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

#%%
def classification_network_training( X, y_one_hot_encoded, 
                                    neuron_count, epochs, learning_rate,
                                    hidden_layers):
    # Initialization
    weights_matrix, bias_matrix = initialization(neuron_count)
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

#%%
classification_model = classification_network_training(X_train, y_train, neuron_count, epochs, learning_rate, hidden_layers)

#%%
predictions = classification_prediction(X_test,
                                        classification_model,
                                        hidden_layers)

actual_values = np.argmax(y_test, axis=1)
reverse_class_encodings = {v: k for k, v in class_encodings.items()}

decoded_predictions = np.vectorize(reverse_class_encodings.get)(predictions)
decoded_actuals = np.vectorize(reverse_class_encodings.get)(actual_values)

#%%
accuracy = accuracy_score(actual_values, predictions)
print(f"Accuracy: {accuracy:.2f}")

conf_matrix = confusion_matrix(actual_values, predictions)
print("Confusion matrix:\n", conf_matrix)
