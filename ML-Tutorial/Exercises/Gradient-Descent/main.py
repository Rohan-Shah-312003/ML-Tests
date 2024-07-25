import numpy as np
import pandas as pd

def gradient_descent(x,y):
  m_curr = b_curr = 0
  learning_rate = 0.0002
  iterations = 50
  n = len(x)

  for i in range(iterations):
    y_pred = m_curr * x + b_curr
    cost = (1/n) * sum([val**2 for val in (y-y_pred)])
    m_d = - (2/n) * sum(x * (y - y_pred))
    b_d = - (2/n) * sum(y - y_pred)
    m_curr = m_curr - learning_rate * m_d
    b_curr = b_curr - learning_rate * b_d

    print("m: {}, b: {}, cost: {}, iterations: {}".format(m_curr, b_curr, cost, i))

df = pd.read_csv("ML-Tutorial/Exercises/Gradient-Descent/marks.csv")

x = np.array((df.math))
y = np.array((df.cs))

gradient_descent(x,y)