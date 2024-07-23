import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math

df = pd.read_csv("ML-Tutorial/Linear-Regression-Multi-Variable/Homeprices.csv")

# print(df)

median_bedrooms = math.floor(df.bedrooms.median())

# print(median_bedrooms)


# replacing the missing values with median values
df.bedrooms = df.bedrooms.fillna(median_bedrooms)

# print(df)


# importing linear regression model and fitting the data to the model

model = LinearRegression()
model.fit(df[['area', 'bedrooms', 'age']], df.price)

print(model.predict([[3000, 3, 40]]))