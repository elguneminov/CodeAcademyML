import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# Load and explore the dataset

digits = datasets.load_digits()
#print(digits)
#print(digits.DESCR)
#print(digits.data)
#print(digits.target)

# Visualising the data

plt.gray() 
plt.matshow(digits.images[100])
plt.show()
#print(digits.target[100])

# K-Means Clustering

model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)

# Visualizing after K-Means

fig = plt.figure(figsize=(8, 3))

fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)
  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()

# Testing Your Model

new_samples = np.array([
[0.00,0.23,2.36,3.05,2.97,0.69,0.00,0.00,0.00,6.33,7.62,7.62,7.62,5.95,0.00,0.00,0.00,5.65,3.06,0.23,4.88,6.86,0.00,0.00,0.00,0.00,0.00,0.08,7.02,5.26,0.00,0.00,0.00,0.00,2.21,5.57,7.62,2.67,0.00,0.00,0.00,5.19,7.63,7.62,7.47,6.10,4.58,0.00,0.00,5.65,6.10,6.10,5.79,5.34,4.35,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.98,3.74,2.29,0.00,0.00,0.00,0.00,3.89,7.62,7.62,7.25,1.37,0.00,0.00,0.00,6.10,5.64,1.15,6.94,5.87,0.00,0.00,0.00,6.02,6.64,2.90,6.86,5.64,0.00,0.00,0.00,2.14,7.17,7.62,7.32,1.91,0.00,0.00,0.00,0.00,0.23,0.76,0.38,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.38,5.34,7.63,6.86,2.90,0.00,0.00,0.00,1.15,7.48,4.35,6.86,6.86,0.00,0.00,0.00,0.00,0.00,4.81,7.62,6.48,0.00,0.00,0.00,0.00,0.00,5.65,6.79,7.62,1.76,0.00,0.00,0.00,1.91,4.04,6.86,7.62,2.06,0.00,0.00,0.00,7.40,7.55,5.65,2.59,0.00,0.00,0.00,0.00,0.54,0.54,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,7.17,3.13,2.21,4.58,0.23,0.00,0.00,0.00,6.56,4.88,2.60,7.63,4.12,0.00,0.00,0.00,5.64,7.40,6.48,7.40,7.32,6.10,2.59,0.00,2.29,4.57,4.58,5.64,7.62,5.26,1.83,0.00,0.00,0.00,0.00,2.75,7.62,0.76,0.00,0.00,0.00,0.00,0.00,1.68,4.88,0.08,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00]
])

new_labels = model.predict(new_samples)
print(new_labels)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')