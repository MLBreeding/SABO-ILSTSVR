import numpy as np

from numpy import linalg


def Hyperplane_2(H,Y,C2,C4,Epsi2,Kmax,Epsi0):
    r = H.shape[0]
    bt = np.random.rand(r)
    bt2 = np.zeros(r)
    k = 1

    while True:
        HtH = np.dot(H.T, H)
        C3_bt2 = C4 * np.diag(bt).T * np.diag(bt)
        Hte = np.dot(H.T, np.ones((r, 1)))
        etH = Hte.T
        ete = np.dot(np.ones((r, 1)).T, np.ones((r, 1)))
        # ete = r
        matrix1 = (1 + C2) * HtH + C3_bt2
        matrix2 = (1 + C2) * Hte
        matrix3 = (1 + C2) * etH
        matrix4 = (1 + C2) * ete + C4
        matrixL = np.vstack([np.hstack([matrix1, matrix2]), np.hstack([matrix3, matrix4])])
        HtY = np.dot(H.T, Y)
        etY = np.dot(np.ones((r, 1)).T, Y)
        matrix5 = (1 + C2) * HtY + (1 - C2) * Hte * Epsi2
        matrix6 = (1 + C2) * etY + (1 - C2) * ete * Epsi2
        matrixR = np.vstack([matrix5, matrix6])

        u2 = np.dot(linalg.inv(matrixL), matrixR)
        w2 = u2[:len(u2)-1]
        b2 = u2[len(u2)-1]

        bt2 = bt
        for i in range(r):
            up = 1
            down = np.abs(w2[i]) + Epsi0
            bt[i] = np.sqrt(up / down)

        k = k + 1
        if k > Kmax or np.dot((bt2 - bt).T, bt2 - bt) < Epsi0:
            break
        
    return [w2, b2]
