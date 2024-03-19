import numpy as np
from sklearn.model_selection import train_test_split,KFold
from ILSTSVR import *
import scipy
import pandas as pd
from sklearn.metrics import mean_squared_error
from Find_param import find_param


# X = pd.read_csv("C:/.../.../genotype.csv", dtype=np.float32)
# y = pd.read_csv("C:/.../.../phenotype.csv", dtype=np.float32)

X = np.array(X)
y = np.array(y)

r2 = 0
mse = 0
pearson = 0

flag = True

folds = KFold(n_splits=10, shuffle=True)

for fold_, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    if flag == True:
        best_param = find_param(X[train_idx, :], y[train_idx, :])
        flag = False
    clf = ILSTSVR(C1 = best_param[0], C2 = best_param[0], C3 = best_param[1], C4 = best_param[1], kernel_param = best_param[2])
        
    clf.fit(X[train_idx, :], y[train_idx, :])
    y_pred = clf.predict(X[val_idx, :]).reshape(-1)
    y_test = y[val_idx, :].reshape(-1)

    mse += mean_squared_error(y_test, y_pred) / folds.n_splits
    pearson += scipy.stats.pearsonr(y_test, y_pred)[0] / folds.n_splits

print("MSE: ", mse)
print("person: ", pearson)
print(best_param)
