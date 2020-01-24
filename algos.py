import numpy as np

def tabular_vi(T, R, g, steps=np.inf, eps=1e-9):
    V = np.random.random_sample(T.shape[1])
    Q = np.random.random_sample(T.shape[0:2])
    done = False
    while steps and not done:
        delta = 0
        V_old = V
        Q_old = Q
        # Q = np.einsum('ijk,ijk->ij', T, R) + g*np.einsum('ijk,k->ij', T, V)
        Q = R + g*np.einsum('ijk,k->ij', T, V)
        V = Q.max(axis=0)
        delta = max(delta, np.linalg.norm(Q_old - Q))
        if delta < eps:
            done = True
        steps -= 1
    return Q, V
