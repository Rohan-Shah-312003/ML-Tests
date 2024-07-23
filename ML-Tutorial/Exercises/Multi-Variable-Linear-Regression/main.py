import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math
from word2number import w2n

# importing dataset as dataframe

df = pd.read_csv('ML-Tutorial/Exercises/Multi-Variable-Linear-Regression/hiring.csv')

# print(df)


# replacing empty cells with elements

df.experience = df.experience.fillna("zero")
df.test_score = df.test_score.fillna(math.floor(df.test_score.median()))

# print(df)

df.experience = df.experience.apply(w2n.word_to_num)

# print(df)


model = LinearRegression()
model.fit(df[['experience', 'test_score', 'interview_score']], df.salary)

print(model.predict([[2, 9, 6]]))
print(model.predict([[12, 10, 10]]))
