import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

import numpy as np
import matplotlib.pyplot as plt


def make_meshgrid(ax, h=.02):
    # x_min, x_max = x.min() - 1, x.max() + 1
    # y_min, y_max = y.min() - 1, y.max() + 1
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def draw_boundary(ax, clf):

    xx, yy = make_meshgrid(ax)
    return plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.5)

## AARON JUDGE

## Become familiar with the data

#print(aaron_judge.columns)
#print(aaron_judge.description.unique())
#print(aaron_judge.type.unique())

## Clean and prep the data for analysis

aaron_judge['type'] = aaron_judge['type'].map({'S':1, 'B':0})
#print(aaron_judge.type)
#print(aaron_judge['plate_x'])

aaron_judge = aaron_judge.dropna(subset = ['plate_x', 'plate_z','type'])

## Plotting the pitches

fig, ax = plt.subplots()
plt.scatter(aaron_judge.plate_x, aaron_judge.plate_z, c=aaron_judge.type, cmap = plt.cm.coolwarm, alpha=0.25)
plt.xlabel('How far left or right the pitch is from the center of home plate')
plt.ylabel('How high off the ground the pitch was')
plt.title('Graph where the strikes are red and the balls blue')

## Building the SVM for AAron Judge
training_set, validation_set, = train_test_split(aaron_judge, random_state = 1)

classifier = SVC(kernel = 'rbf', gamma=100, C=100)
classifier.fit(training_set[['plate_x', 'plate_z']], training_set[['type']])
draw_boundary(ax, classifier)

#print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set[['type']]))

plt.show()
plt.clf()

## Optimising the SVM

for i in range(1, 11, 3):
    for n in range(1, 11, 3):
        classifier = SVC(kernel = 'rbf', gamma=i, C=n)
        classifier.fit(training_set[['plate_x', 'plate_z']], training_set[['type']])
        score = classifier.score(validation_set[['plate_x', 'plate_z']], validation_set[['type']])
        print('For gamma = {g} and C = {c}, score is {s}'.format(g=i, c=n, s=score))

# Best score appears to be For gamma = 1 and C = 16, score is 0.8355957767722474

## DAVID ORTIZ

## Clean and prep the data for analysis

david_ortiz['type'] = david_ortiz['type'].map({'S':1, 'B':0})
david_ortiz = david_ortiz.dropna(subset = ['plate_x', 'plate_z','type'])

## Plotting the pitches

fig2, ax2 = plt.subplots()
plt.scatter(david_ortiz.plate_x, david_ortiz.plate_z, c=david_ortiz.type, cmap = plt.cm.coolwarm, alpha=0.25)
plt.xlabel('How far left or right the pitch is from the center of home plate')
plt.ylabel('How high off the ground the pitch was')
plt.title('Graph where the strikes are red and the balls blue')

## Building the SVM for AAron Judge
training_set2, validation_set2, = train_test_split(david_ortiz, random_state = 1)

classifier2 = SVC(kernel = 'rbf', gamma=100, C=100)
classifier2.fit(training_set2[['plate_x', 'plate_z']], training_set2[['type']])
draw_boundary(ax2, classifier2)

print(classifier2.score(validation_set2[['plate_x', 'plate_z']], validation_set2[['type']]))

plt.show()