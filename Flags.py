import codecademylib3_seaborn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Project Aim: Predict where a country is based only on the colors of its flag

# Load and investigate the data

flags = pd.read_csv('flags.csv', header=0)
#print(flags.head())
#print(flags.columns)

# Creating Your Data and Labels

labels = flags[['Landmass']]
data = flags[["Red", "Green", "Blue", "Gold", "White", "Black", "Orange"]]

training_data, test_data, training_labels, test_labels = train_test_split(data, labels, random_state=1)

# Make and Test the Model

tree = DecisionTreeClassifier(random_state=1)
tree.fit(training_data, training_labels)
#print(tree.score(test_data, test_labels))

# Optimising the model by pruning the tree
list_of_scores = []
for i in range(1, 21):
  tree = DecisionTreeClassifier(random_state=1, max_depth = i)
  tree.fit(training_data, training_labels)
  score = tree.score(test_data, test_labels)
  list_of_scores.append(score)

plt.plot(range(1,21), list_of_scores)
plt.xlabel('Max Depth of Tree')
plt.ylabel('Score of Test Data')
plt.show()

# Optimising the model by adding more features

data2 = flags[["Red", "Green", "Blue", "Gold",  "White", "Black", "Orange",  "Circles", "Crosses","Saltires","Quarters","Sunstars",
"Crescent","Triangle"]]

training_data2, test_data2, training_labels2, test_labels2 = train_test_split(data2, labels, random_state=1)

list_of_scores2 = []
for i in range(1, 21):
  tree2 = DecisionTreeClassifier(random_state=1, max_depth = i)
  tree2.fit(training_data2, training_labels2)
  score2 = tree2.score(test_data2, test_labels2)
  list_of_scores2.append(score2)

plt.clf()

plt.plot(range(1,21), list_of_scores2)
plt.xlabel('Max Depth of Tree')
plt.ylabel('Score of Test Data')
plt.show()