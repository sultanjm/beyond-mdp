# generate random ergodic mdps

import numpy as np
import itertools

def random_mdp(num_a, num_s, Q=None, g=0.999, normalize=False, epsilon=1e-3):
    # T = np.random.random_sample([num_a, num_s, num_s]) + epsilon
    T = np.random.random_sample([num_a, num_s, num_s]) + epsilon
    T = T/T.sum(axis=2,keepdims=True)
    if Q is None:
        R = np.random.random_sample([num_a,num_s,num_s])
    else:
        V = Q.max(axis=0)
        R = Q - g*np.einsum('ijk,k->ij', T, V)
        # we assume R[a][s] instead of R[a][s][s'] to avoid the inverse
        # Tinv = np.linalg.inv(T)
        # R = np.einsum('ijk,ij->ijk',Tinv,R)
    rmin = R.min()
    rmax = R.max()
    if normalize:
        n = 1/(rmax-rmin)
        R = n*(R - rmin)
    return T,R,rmin,rmax

def stationary_dist(T, Pi=None, d=None, steps=np.inf, eps=1e-9):
    if d is None:
        d = np.random.random_sample(T.shape[1])
        d = d/d.sum()
    if Pi is not None:
        T = np.einsum('ijk,ji->jk',T,Pi)
    done = False
    while steps and not done:
        delta = 0
        d_old = d
        d = np.einsum('ij,i->j',T,d)
        delta = max(delta, np.linalg.norm(d_old - d))
        if delta < eps:
            done = True
        steps -= 1
    return d

def surrogate_mdp(T,R,phi,B,states):
    num_x = len(states)
    num_a, num_s = R.shape
    T_mdp = np.zeros([num_a,num_x,num_x])
    pre_T_mdp = np.zeros([num_a,num_s,num_x])
    R_mdp = np.zeros([num_a,num_x])
    W = np.zeros([num_a,num_x])

    for a,s in itertools.product(range(num_a), range(num_s)):
        x = states.index(phi[s])
        for s_nxt in range(num_s):
            x_nxt = states.index(phi[s_nxt])
            pre_T_mdp[a][s][x_nxt] = pre_T_mdp[a][s][x_nxt] + T[a][s][s_nxt]
        R_mdp[a][x] = R_mdp[a][x] + B[a][s]*R[a][s]
        T_mdp[a][x] = T_mdp[a][x] + B[a][s]*pre_T_mdp[a][s]
        W[a][x] = W[a][x] + B[a][s]
    R_mdp = R_mdp / W
    T_mdp = T_mdp / T_mdp.sum(axis=2,keepdims=True)
    return T_mdp,R_mdp

def greedy_policy(Q, epsilon=0.0):
    num_a,num_s = Q.shape
    pi = np.zeros([num_s,num_a]) + epsilon/num_a
    idx = Q.argmax(axis=0)
    for s in range(num_s):
        pi[s][idx[s]] = pi[s][idx[s]] + 1
    pi = pi/pi.sum(axis=1,keepdims=True)
    return pi