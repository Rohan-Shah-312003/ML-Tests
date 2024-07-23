import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings


df = pd.read_csv("ML-Tutorial/Linear-Regression-Single-Variable/homeprices.csv")


# plt.xlabel("Area")
# plt.ylabel("Price")
# plt.scatter(df.area, df.price, color="red", marker="+")
# plt.show()

reg = LinearRegression()
reg.fit(df[['area']], df.price)


with warnings.catch_warnings(): # to ignore a warning shown for predicting the fitting
    warnings.filterwarnings('ignore') # to ignore a warning shown for predicting the fitting
    
    print(reg.predict([[3300]]))

    print(reg.coef_)

    print(reg.intercept_)


# plt.xlabel("Area")
# plt.ylabel("Price")
# plt.scatter(df.area, df.price, color="red", marker="+")
# plt.plot(df.area, reg.predict(df[['area']]), color="blue")
# plt.show()

# df2 = pd.read_csv("ML-Tutorial/Linear-Regression-Single-Variable/areas.csv")

# print(df2.head(3))

# p = reg.predict(df2)
# df2['price'] = p
# df2.to_csv("ML-Tutorial/Linear-Regression-Single-Variable/prediction.csv", index=False)