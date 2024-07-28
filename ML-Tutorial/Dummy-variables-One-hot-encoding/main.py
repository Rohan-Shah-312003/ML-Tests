import pandas as pd

df = pd.read_csv("ML-Tutorial/Dummy-variables-One-hot-encoding/homeprices.csv")
print(df)

dummies = pd.get_dummies(df.town)

merged_df = pd.concat([df, dummies], axis='columns')
print(merged_df)

