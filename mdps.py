# generate random ergodic mdps

import numpy as np
import scipy.linalg as la
import itertools

def random_mdp(num_a, num_s, args, Q=None):
    T = random_sample([num_a, num_s, num_s]) + args.eps
    T = T/T.sum(axis=2,keepdims=True)
    R, rmin, rmax = reward_matrix(num_a, num_s, T, args, Q)
    return T,R,Q,rmin,rmax

def random_skewed_mdp(num_a, num_s, args, Q=None):
    idx = np.random.choice(range(num_s), num_a * num_s)
    idx2 = [num_s*x+idx[x] for x in range(len(idx))]
    T = np.zeros(num_a*num_s*num_s) + args.eps
    T[idx2] = 1
    T = T.reshape([num_a, num_s, num_s])
    T = T/T.sum(axis=2,keepdims=True)
    R, rmin, rmax = reward_matrix(num_a, num_s, T, args, Q)
    return T,R,Q,rmin,rmax

def reward_matrix(num_a, num_s, T, args, Q=None):
    if Q is None:
        R = random_sample([num_a,num_s])
    else:
        V = Q.max(axis=0)
        R = Q - args.g*np.einsum('ijk,k->ij', T, V)
        # we assume R[a][s] instead of R[a][s][s'] to avoid the inverse
        # Tinv = np.linalg.inv(T)
        # R = np.einsum('ijk,ij->ijk',Tinv,R)
    rmin = R.min()
    rmax = R.max()
    # normalize rewards
    if args.n:
        if np.isclose(rmax, rmin):
            R = R/rmax
        else:
            n = 1/(rmax-rmin)
            R = n*(R - rmin)
    return R,rmin,rmax

def stationary_dist(T, args, Pi=None, d=None):
    if d is None:
        d = np.random.random_sample(T.shape[1])
        d = d/d.sum()
    if Pi is not None:
        T = np.einsum('ijk,ji->jk',T,Pi)
    done = False
    steps = args.steps
    while steps and not done:
        delta = 0
        d_old = d
        d = np.einsum('ij,i->j',T,d)
        delta = max(delta, np.linalg.norm(d_old - d))
        if delta < args.eps:
            done = True
        steps -= 1
    return d

def stationary_dist_eigenvalue(T, args, Pi=None, d=None):
    if d is None:
        d = np.random.random_sample(T.shape[1])
        d = d/d.sum()
    if Pi is not None:
        T = np.einsum('ijk,ji->jk',T,Pi)
    w, vl, _ = la.eig(T, left=True)
    #w, vl, _ = la.eig(T.T, left=True)
    idx = np.where(np.isclose(w,1))
    for i in idx[0]:
        d = vl[:,i]/vl[:,i].sum()
        if np.alltrue(d >= 0) and np.isclose(d.sum(), 1):
            break
    return d.real

def stationary_dist_cesaro(T, args, Pi=None, d=None):
    if d is None:
        d = np.random.random_sample(T.shape[1])
        d = d/d.sum()
    if Pi is not None:
        T = np.einsum('ijk,ji->jk',T,Pi)
    done = False
    t = 1
    old_avg = np.zeros(T.shape[1])
    steps = args.steps
    while steps and not done:
        delta = 0
        d = np.einsum('ij,i->j',T,d)
        new_avg = old_avg + (1/t)*(d - old_avg)
        t += 1
        delta = max(delta, np.linalg.norm(old_avg - new_avg))
        old_avg = new_avg
        if delta < args.eps:
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

def greedy_policy(Q, args):
    num_a,num_s = Q.shape
    pi = np.zeros([num_s,num_a]) + args.greedy_eps/num_a
    idx = Q.argmax(axis=0)
    for s in range(num_s):
        pi[s][idx[s]] = pi[s][idx[s]] + 1
    pi = pi/pi.sum(axis=1,keepdims=True)
    return pi

def random_policy(num_a, num_s, args):
    # uniform random policy
    if args.u:
        pi = np.random.random_sample([num_s,num_a]) + args.eps
    else:
        pi = random_sample([num_s,num_a]) + args.eps
    pi = pi/pi.sum(axis=1,keepdims=True)
    return pi

def fixed_policy(num_a, num_s, the_action=0):
    pi = np.zeros([num_s,num_a])
    pi[:, the_action] = 1
    return pi

def greedy_policy_all(Q, args):
    mx = Q.max(axis=0)
    pi = np.isclose(Q.T, mx[:,np.newaxis])
    pi = pi/pi.sum(axis=1,keepdims=True)
    return pi

def common_support_policies(pi_1, pi_2):
    same = True
    if 0 in np.logical_and(pi_1>0,pi_2>0).sum(axis=1):
        same = False
    return same

def same_optimal_policies(pi_base, pi):
    same = False
    if common_support_policies(pi_base, pi):
        pi_base_neg = np.isclose(pi_base, 0.0)
        if not 1 in np.logical_and(pi_base_neg,pi>0).sum(axis=1):
            same = True
    return same

def random_va_mdp(num_a, num_s, max_num_x, args, Q_va=None):
    if Q_va is None:
        Q_va = random_sample([num_a, max_num_x])
    states = list(set(va_states(Q_va,args)))
    # generate a Q-function with set_x va-states only
    Q = va_states_repeat(num_a, num_s, states, args)
    # generate an random mdp with Q as the Q-function
    T,R,rmin,rmax = random_skewed_mdp(num_a, num_s, args, Q)
    return T,R,Q,rmin,rmax
    
def va_states(Q, args):
    opt_acts = pack_opt_acts(Q, args)
    return zip(Q.max(axis=0) // args.va_eps, opt_acts)

def pack_opt_acts(Q, args):
    # if all optimal actions aggregated
    if args.a:
        V_mask = np.isclose(Q.T, Q.max(axis=0)[:,np.newaxis])
    else:
        V_mask = Q.argmax(axis=0)[:,None] == range(Q.shape[0])
    return [V_mask[s].dot(1 << np.arange(V_mask[s].size)[::-1]) for s in range(Q.shape[1])]        


def unpack_opt_acts(dec_act):
    return np.fromstring(np.binary_repr(dec_act), dtype='S1').astype(int)

def va_states_repeat(num_a, num_s, states, args):
    num_x = len(states)
    # get num_x random fractions of num_s
    splits = sorted(np.random.choice(range(1,num_s), num_x-1))
    prop_x = np.array([a - b for a, b in zip(splits + [num_s], [0] + splits)])
    # for each fraction generate random samples with lower than v value at a
    Q = np.array([])
    for x in range(num_x):
        if prop_x[x] != 0:
            Qx = states[x][0]*args.va_eps*random_sample([num_a, prop_x[x]])
            acts = unpack_opt_acts(states[x][1])
            Qx[np.where(acts == 1),:] = (states[x][0] + 0.5)*args.va_eps
            if len(Q) != 0:
                Q = np.concatenate((Q, Qx), axis=1)
            else:
                Q = Qx
    return Q


def random_sample(size):
    levels = 2
    return np.random.randint(levels, size=size) / (levels - 1)
    # return np.random.beta(0.5,0.5, size=size)
    # return np.random.random_sample(size)