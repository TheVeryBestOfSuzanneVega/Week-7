import csv
import numpy as np
import pandas as pd
import sys

##variables
alpha = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.75]
b = np.array([[0],[0],[0]])
regr = []
r = 0
old_r = 0

#preprocessing
df = pd.read_csv(sys.argv[1], header = None, names=["age","weight","height"])
df["age"] = (df["age"] - df["age"].mean()) / df["age"].std()
df["weight"] = (df["weight"] - df["weight"].mean()) / df["weight"].std()
df.insert(0, "intercept", 1)
X = np.array(df.as_matrix(columns=["intercept", "age","weight"]))
Y = np.array(df.as_matrix(columns=["height"]))
n = len(df.index)

##linear function
#def f(x):
#    return b[0]+b[1]*x+b[2]*x

## GRADIENT DESCENT
for a in alpha:
    b = np.array([[0],[0],[0]])
    b_new = np.copy(b)
    for j in range(100):
        b_new = b_new - a*(1/n)*(np.transpose(X)@(X@b - Y))
        b = np.copy(b_new)

    regr.append([a,j+1] + np.ndarray.tolist(np.ndarray.flatten(b)))

with open(sys.argv[2], 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    for w in regr:
        wr.writerow(w)
