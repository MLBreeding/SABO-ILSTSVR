from ILSTSVR import *
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_squared_error
from SABO import SABO

def find_param(X, y):
    n, epochs, lo, hi, dim = 20, 10, [0.1, 0.1, np.sqrt(X.shape[1]) - 20], [10, 10, np.sqrt(X.shape[1]) + 20], 3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    def fitness(param, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test):
        clf = ILSTSVR(C1 = param[0], C2 = param[0], C3 = param[1], C4 = param[1], kernel_param = param[2])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test).reshape(-1)
        y_test = y_test.reshape(-1)

        mse = mean_squared_error(y_test, y_pred)

        return mse
    
    Best_score, Best_pos = SABO(n, epochs, lo, hi, dim, fitness)
 
    return Best_pos