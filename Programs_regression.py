import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def calculate_weights(point, features_matrix, bandwidth):
    num_samples, num_features = np.shape(features_matrix)
    weights = np.asmatrix(np.eye(num_samples))  # Identity matrix
    for j in range(num_samples):
        difference = point - features_matrix[j]
        weights[j, j] = np.exp(difference * difference.T / (-2.0 * bandwidth ** 2))
    return weights

def compute_local_weight(point, features_matrix, target_matrix, bandwidth):
    weights = calculate_weights(point, features_matrix, bandwidth)
    weighted_matrix = (features_matrix.T * (weights * features_matrix)).I * (features_matrix.T * (weights * target_matrix.T))
    return weighted_matrix

def local_weighted_regression(features_matrix, target_matrix, bandwidth):
    num_samples, num_features = np.shape(features_matrix)
    predicted_values = np.zeros(num_samples)
    for i in range(num_samples):
        predicted_values[i] = features_matrix[i] * compute_local_weight(features_matrix[i], features_matrix, target_matrix, bandwidth)
    return predicted_values

def plot_graph(features_matrix, predicted_values):
    sorted_index = features_matrix[:, 1].argsort(0)
    sorted_features = features_matrix[sorted_index][:, 0]
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(total_bill, tip_amount, color='green')
    ax.plot(sorted_features[:, 1], predicted_values[sorted_index], color='red', linewidth=5)
    plt.xlabel('Total Bill')
    plt.ylabel('Tip Amount')
    plt.show()

# Load the dataset
data = pd.read_csv("bill.csv")
total_bill = np.array(data.total_bill)
tip_amount = np.array(data.tip)

# Prepare the feature and target matrices
matrix_total_bill = np.asmatrix(total_bill)
matrix_tip = np.asmatrix(tip_amount)
num_samples = np.shape(matrix_total_bill)[1]
ones_column = np.asmatrix(np.ones(num_samples))
features_matrix = np.hstack((ones_column.T, matrix_total_bill.T))

# Perform local weighted regression
predicted_tip = local_weighted_regression(features_matrix, matrix_tip, 3)

# Plot the results
plot_graph(features_matrix, predicted_tip)
