import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv")
print(df.head())

# graphing the total production of honey per year
prod_per_year = df.groupby('year').totalprod.mean().reset_index()
#print(prod_per_year)

X = prod_per_year.year.values.reshape(-1, 1)
y = prod_per_year.totalprod

plt.plot(X, y, "o")
plt.title('Total production of Honey per year')
plt.xlabel('Year')
plt.ylabel('Total Production')

# Creating the linear regression model
regr = linear_model.LinearRegression()
regr.fit(X, y)
slope = regr.coef_
intercept = regr.intercept_
y_predict = regr.predict(X)

plt.plot(X, y_predict, "*")
plt.show()

# Predicting the Honey Decline
X_future = np.array(range(2013, 2051)).reshape(-1, 1)

future_predict = regr.predict(X_future)

plt.plot(X_future, future_predict, "s")
plt.show()