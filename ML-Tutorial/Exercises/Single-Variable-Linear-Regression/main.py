import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv("ML-Tutorial/Exercises/Single-Variable-Linear-Regression/canada_areas.csv")

# plotting the imported data

# plt.xlabel("Year")
# plt.ylabel("Income")
# plt.scatter(df.year, df.income, color="red", marker="+")
# plt.show()

model = LinearRegression()

model.fit(df[['year']], df.income)

print(model.predict([[2020]]))