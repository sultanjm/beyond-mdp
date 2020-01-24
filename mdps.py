# generate random ergodic mdps

import numpy as np

def random_mdp(num_a, num_s, Q=None, g=0.999, epsilon=1e-3):
    T = np.random.random_sample([num_a, num_s, num_s]) + epsilon
    # T = np.array([[[1,0],[1,0]],[[0,1],[0,1]]])
    T = T/T.sum(axis=2,keepdims=True)
    if Q is None:
        R = np.random.random_sample([num_a,num_s,num_s])
    else:
        V = Q.max(axis=0)
        R = Q - g*np.einsum('ijk,k->ij', T, V)
        # although we made sure it is invertable, it is a numerical nightmare
        # maybe, we can check the condition number
        # we assume R[a][s] instead of R[a][s][s'] to avoid the inverse
        # Tinv = np.linalg.inv(T)
        # R = np.einsum('ijk,ij->ijk',Tinv,R)

    rmin = R.min()
    rmax = R.max()
    n = 1/(rmax-rmin)
    R = n*(R - rmin)

    return T,R,rmin,rmax