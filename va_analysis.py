import mdps
import algos

import numpy as np

def va_analysis(num_s=1000, num_a=2, g=0.999, va_eps=1e-2, eps=1e-9):
    # set a Q-function
    Q = np.random.random_sample([num_a,num_s])
    # get a random (T,R,rmin,rmax) for this Q
    T,R,rmin,rmax = mdps.random_mdp(num_a,num_s,Q,g)
    # normalize Q with rmin and rmax
    Q = (Q - rmin/(1-g))/(rmax-rmin)
    # do VA abstraction of T based on Q
    phi = list(zip(Q.max(axis=0) // va_eps, Q.argmax(axis=0)))
    va_states = list(set(phi))
    num_x = len(va_states)
    # use ANY STATIONARY VA-STATE BASED explorative policy to get a stationary distribution on T
    pi_x = np.random.random_sample([len(va_states), num_a]) + eps
    pi_x = pi_x/pi_x.sum(axis=1,keepdims=True)
    pi_behavior = np.array([pi_x[va_states.index(x)] for x in phi])
    d = mdps.stationary_dist(T, pi_behavior)
    B = [d for a in range(num_a)]
    # use d and phi to get a surrogate MDP (T_mdp,R_mdp)
    T_mdp,R_mdp = mdps.surrogate_mdp(T, R, phi, B, va_states)
    # get the optimal policy on this surrogate MDP using VI
    Q_mdp = algos.tabular_vi(T_mdp,R_mdp,g)
    pi_mdp = np.zeros([num_x,num_a])
    idx = Q_mdp.argmax(axis=0)
    for x in range(num_x):
        pi_mdp[x][idx[x]] = 1
    # use this optimal policy to get values of (T,R)
    pi_via_mdp = np.array([pi_mdp[va_states.index(x)] for x in phi])
    Q_via_mdp = algos.tabular_vi(T, R, g, pi_via_mdp)
    # compare these values with the normalized Q
    print(np.linalg.norm(Q-Q_via_mdp))
    
va_analysis()