from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()

# Becoming familiar with the data
# Note: This data is already normalised

print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)

print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

# Splitting the data into Training and Validation Sets

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

print(len(training_data), len(training_labels))

# Running the classifier

classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))

# Finding the Optimal k value

list_of_scores = []
for i in range (1, 101):
  classifier = KNeighborsClassifier(n_neighbors = i)
  classifier.fit(training_data, training_labels)
  list_of_scores.append([classifier.score(validation_data, validation_labels), i])

maximum_score = max(list_of_scores)
print('The optimal k is {} with a score of {}.'.format(maximum_score[1], maximum_score[0]))

# Graphing the Optimal k value

k_list = range (1, 101)
accuracies = []
for i in range (1, 101):
  classifier = KNeighborsClassifier(n_neighbors = i)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data, validation_labels))

plt.plot(k_list, accuracies)
plt.xlabel('K Value')
plt.ylabel('Accuracy Score')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()
