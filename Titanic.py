import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## Load the passenger data
passengers = pd.read_csv('passengers.csv')
#print(passengers.info())

## Clean and prep data for analysis

# Update sex column to numerical
passengers['Sex'] = passengers.Sex.apply(lambda x: 1 if x == 'female' else 0)
#print(passengers.head())

# Fill the nan values in the age column
#print(passengers['Age'].values)
passenger_age_mean = passengers.Age.mean()
#print(passenger_age_mean)
passengers.Age.fillna(value =passenger_age_mean, inplace=True)
#print(passengers['Age'].values)

# Create a first class column
passengers['FirstClass'] = passengers.Pclass.apply(lambda x: 1 if x == 1 else 0)
#print(passengers.head())

# Create a second class column
passengers['SecondClass'] = passengers.Pclass.apply(lambda x: 1 if x == 2 else 0)
#print(passengers.head())

# Create a third class column
passengers['ThirdClass'] = passengers.Pclass.apply(lambda x: 1 if x == 3 else 0)
#print(passengers.head())

## Select and Split the data

# Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers[['Survived']]

# Perform train, test, split
features_train, features_test, survival_train, survival_test = train_test_split(features, survival, train_size=0.8, test_size=0.2)

## Normalise the data

# Scale the feature data so it has mean = 0 and standard deviation = 1
normalise = StandardScaler()
features_train = normalise.fit_transform(features_train)
features_test = normalise.transform(features_test)

# Create and train the model
model = LogisticRegression()
model.fit(features_train, survival_train)

# Score the model on the train data
print(model.score(features_train, survival_train))

# Score the model on the test data
print(model.score(features_test, survival_test))

# Analyze the coefficients
coefficients = model.coef_
feature_names = ['Sex','Age','FirstClass','SecondClass']
for a,b in list(zip(feature_names, model.coef_[0])):
  print('The coefficient for {} is {}'.format(a, b))

## Predict with the model

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Folu = np.array([1.0,29.0,0.0,0.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, Folu])

# Scale the sample passenger features
sample_passengers = normalise.transform(sample_passengers)
#print(sample_passengers)

# Make survival predictions!
print(model.predict(sample_passengers))
print(model.predict_proba(sample_passengers))