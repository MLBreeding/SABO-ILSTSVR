import numpy as np


def SABO(N, epochs, lo, hi, m, fitness):
    lo = np.ones((1, m)) * lo
    hi = np.ones((1, m)) * hi
    X = np.zeros((N, m))
    fit = np.zeros(N)

    for i in range(m):
        X[:, i] = lo[:, i] + np.random.rand(N, 1) @ (hi[:, i] - lo[:, i])

    for i in range(N):
        L = X[i, :]
        fit[i] = fitness(L)

    for epoch in range(epochs):
        blocation = np.argmin(fit)
        Fbest = fit[blocation]
 
        if epoch == 0:
            xbest = X[blocation, :]
            fbest = Fbest
        elif Fbest < fbest:
            fbest = Fbest
            xbest = X[blocation, :]
        DX = np.zeros((N, m))
        for i in range(N):
            for j in range(N):
                I = np.round(1 + np.random.rand() + np.random.rand())
                for d in range(m):
                    DX[i, d] = DX[i, d] + ((X[j, d] - (I * X[i, d])) * np.sign(fit[i] - fit[j]))

            X_new_P1 = X[i, :] + ((np.random.rand(1, m) @ DX[i, :]) / N)
            X_new_P1 = np.maximum(X_new_P1, lo.reshape(-1))
            X_new_P1 = np.minimum(X_new_P1, hi.reshape(-1))

            L = X_new_P1
            fit_new_P1 = fitness(L)
            if fit_new_P1 < fit[i]:
                X[i, :] = X_new_P1
                fit[i] = fit_new_P1

    Best_score = fbest
    Best_pos = xbest
    return Best_score, Best_pos