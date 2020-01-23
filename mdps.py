# generates random ergodic mdps

import numpy as np

def random_mdp(num_s, num_a, Q=None, g=0.999, epsilon=1e-6):
    T = np.random.random_sample([num_s, num_a, num_s]) + epsilon
    T = T/T.sum(axis=2,keepdims=True)
    if Q is None:
        R = np.random.random_sample([num_s,num_a])
    else:
        V = Q.max(axis=1)
        R = Q - g*np.einsum('ijk,k->ij', T, V)
    return T,R