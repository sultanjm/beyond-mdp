# generate random ergodic mdps

import numpy as np
import itertools

def random_mdp(num_a, num_s, Q=None, g=0.999, normalize=False, epsilon=1e-3):
    T = random_sample([num_a, num_s, num_s]) + epsilon
    T = T/T.sum(axis=2,keepdims=True)
    R, rmin, rmax = reward_matrix(num_a, num_s, T, Q, g, normalize)
    return T,R,rmin,rmax

def random_skewed_mdp(num_a, num_s, Q=None, g=0.999, normalize=False, epsilon=1e-9):
    idx = np.random.choice(range(num_s), num_a * num_s)
    idx2 = [num_s*x+idx[x] for x in range(len(idx))]
    T = np.zeros(num_a*num_s*num_s) + epsilon
    T[idx2] = 1
    T = T.reshape([num_a, num_s, num_s])
    T = T/T.sum(axis=2,keepdims=True)
    R, rmin, rmax = reward_matrix(num_a, num_s, T, Q, g, normalize)
    return T,R,rmin,rmax

def reward_matrix(num_a, num_s, T, Q=None, g=0.999, normalize=False):
    if Q is None:
        R = random_sample([num_a,num_s,num_s])
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
    return R,rmin,rmax

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

def stationary_dist_cesaro(T, Pi=None, d=None, steps=np.inf, eps=1e-6):
    if d is None:
        d = np.random.random_sample(T.shape[1])
        d = d/d.sum()
    if Pi is not None:
        T = np.einsum('ijk,ji->jk',T,Pi)
    done = False
    t = 1
    old_avg = np.zeros(T.shape[1])
    while steps and not done:
        delta = 0
        d = np.einsum('ij,i->j',T,d)
        new_avg = old_avg + (1/t)*(d - old_avg)
        t += 1
        delta = max(delta, np.linalg.norm(old_avg - new_avg))
        old_avg = new_avg
        if delta < eps:
            done = True
        steps -= 1
    return new_avg

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
    R_mdp = np.divide(R_mdp, W, out=np.zeros_like(R_mdp), where=W!=0)
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

def random_policy(num_a, num_s, eps=1e-9):
    pi = random_sample([num_s,num_a]) + eps
    pi = pi/pi.sum(axis=1,keepdims=True)
    return pi

def fixed_policy(num_a, num_s, the_action=0, eps=1e-9):
    pi = np.zeros([num_s,num_a])
    pi[:, the_action] = 1
    return pi

def greedy_policy_all(Q, epsilon=0.0):
    mx = Q.max(axis=0)
    pi = np.isclose(Q.T, mx[:,np.newaxis])
    pi = pi/pi.sum(axis=1,keepdims=True)
    return pi

def same_optimal_policies(pi_1, pi_2):
    same = True
    if 0 in np.logical_and(pi_1>0,pi_2>0).sum(axis=1):
        same = False
    return same

def random_va_mdp(num_a=2, num_s=80, max_num_x=4, va_eps=1e-6, g=0.999, Q_va=None, normalize_rewards=False):
    if Q_va is None:
        Q_va = random_sample([num_a, max_num_x])
    va_states = list(set(zip(Q_va.max(axis=0) // va_eps, Q_va.argmax(axis=0))))
    # generate a Q-function with set_x va-states only
    Q = va_states_repeat(num_a, num_s, va_states, va_eps)
    # generate an random mdp with Q as the Q-function
    T,R,rmin,rmax = random_skewed_mdp(num_a, num_s, Q, g, normalize_rewards)
    return T,R,Q,rmin,rmax,

def va_states_repeat(num_a, num_s, va_states, va_eps=1e-6):
    num_x = len(va_states)
    # get num_x random fractions of num_s
    splits = sorted(np.random.choice(range(1,num_s), num_x-1))
    prop_x = np.array([a - b for a, b in zip(splits + [num_s], [0] + splits)])
    # for each fraction generate random samples with lower than v value at a
    Q = np.array([])
    for x in range(num_x):
        if prop_x[x] != 0:
            Qx = va_states[x][0]*va_eps*random_sample([num_a, prop_x[x]])
            Qx[va_states[x][1],:] = (va_states[x][0] + 0.5)*va_eps
            if len(Q) != 0:
                Q = np.concatenate((Q, Qx), axis=1)
            else:
                Q = Qx
    return Q

def random_sample(size):
    return np.random.randint(8, size=size)
    # return np.random.beta(0.5,0.5, size=size)
    # return np.random.random_sample(size)