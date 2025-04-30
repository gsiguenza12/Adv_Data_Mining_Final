#-------------------------------------------------------------------------
# AUTHORS: Moaz Ali, Gabriel Alfredo Siguenza
# FILENAME: data_mining_project.py
# SPECIFICATION:  
# FOR: CS 5990 - Advanced Data Mining Final Project
#-----------------------------------------------------------*/

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


path = './bots_vs_users.csv'

print("Path to dataset files:", path)

df = pd.read_csv(path)
df.head()

print("\nBasic Info:")
print(df.info)

print("\nMissing Values:")
missing = df.isnull().sum()
print(missing[missing > 0])

print("\nSkewness Check:")
skewed_features = df.select_dtypes(include=['number']).apply(lambda x: x.skew()).sort_values(ascending=False)
print(skewed_features)

print("\nRecommendation:")
for col, skew in skewed_features.items():
    # print("\n")
    if abs(skew) > 1:
        print(f"{col} is highly skewed (skew={skew:.2f}). Suggest: Apply log or sqrt transform.")
    elif abs(skew) > 0.5:
        print(f"{col} is moderately skewed (skew={skew:.2f}). Transform optional.")
    else:
        print(f"{col} is fairly symmetric (skew={skew:.2f}). No action needed.")