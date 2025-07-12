#%%
import sys
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

#%%
root_directory = r"" # Enter directory path here
sys.path.append(root_directory) 
from neural_network_model import regression_network_training, regression_prediction
#%%
dataset = pd.read_csv(root_directory + r"data/BostonHousing.csv")
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