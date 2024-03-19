import numpy as np

from numpy import linalg
from cvxopt import solvers,matrix


def Hyperplane_1(H,Y,C1,C3,Epsi1,Kmax,Epsi0):
    r = H.shape[0]
    bt = np.random.rand(r)
    bt2 = np.zeros(r)
    k = 1

    while True:
        HtH = np.dot(H.T, H)
        C3_bt2 = C3 * np.diag(bt).T * np.diag(bt)
        Hte = np.dot(H.T, np.ones((r, 1)))
        etH = Hte.T
        ete = np.dot(np.ones((r, 1)).T, np.ones((r, 1)))
        # ete = r
        matrix1 = (1 + C1) * HtH + C3_bt2
        matrix2 = (1 + C1) * Hte
        matrix3 = (1 + C1) * etH
        matrix4 = (1 + C1) * ete + C3
        matrixL = np.vstack([np.hstack([matrix1, matrix2]), np.hstack([matrix3, matrix4])])
        HtY = np.dot(H.T, Y)
        etY = np.dot(np.ones((r, 1)).T, Y)
        matrix5 = (1 + C1) * HtY + (C1 - 1) * Hte * Epsi1
        matrix6 = (1 + C1) * etY + (C1 - 1) * ete * Epsi1
        matrixR = np.vstack([matrix5, matrix6])

        u1 = np.dot(linalg.inv(matrixL), matrixR)
        w1 = u1[:len(u1)-1]
        b1 = u1[len(u1)-1]

        bt2 = bt
        for i in range(r):
            up = 1
            down = np.abs(w1[i]) + Epsi0
            bt[i] = np.sqrt(up / down)

        k = k + 1
        if k > Kmax or np.dot((bt2 - bt).T, bt2 - bt) < Epsi0:
            break

    return [w1, b1]