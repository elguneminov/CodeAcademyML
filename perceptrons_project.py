import codecademylib3_seaborn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# AND GATE

# Creating and visualizing AND Data
data = [[0,0], [0, 1], [1, 0], [1, 1]]
labels = [0, 0, 0, 1]
# For XOR Gate change labels to [0,1,1,0], note this data set is not linearly separable
# For OR Gate change labels to [0,1,1,1]
plt.scatter([point[0] for point in data], [point[1] for point in data], c=labels)
plt.show()

# Building the Perceptron
classifier = Perceptron(max_iter=40)
classifier.fit(data, labels)
print(classifier.score(data, labels))

# Visualising the Perceptron
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)
point_grid = list(product(x_values, y_values))
distances = classifier.decision_function(point_grid)
abs_distances = [abs(distance) for distance in distances]
distances_matrix = np.reshape(abs_distances, (100, 100))
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
plt.colorbar(heatmap)
plt.show()
# The purple line is where the distances are 0, that’s the decision boundary!