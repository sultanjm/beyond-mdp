import mdps
import algos

import numpy as np

def va_analysis(num_s=100, num_a=16, g=0.999, va_eps=1e-1, eps=1e-9):
    # set a Q-function
    #Q = np.random.random_sample([num_a,num_s])
    Q = np.random.beta(0.5,0.5,[num_a,num_s])
    # get a random (T,R,rmin,rmax) for this Q
    #T,R,rmin,rmax = mdps.random_mdp(num_a,num_s,Q,g,True)
    # normalize Q with rmin and rmax
    #Q = (Q - rmin/(1-g))/(rmax-rmin)
    T,R,rmin,rmax = mdps.random_mdp(num_a,num_s,Q,g)
    # get the optimal policy
    pi = mdps.greedy_policy(Q)
    # do VA abstraction of T based on Q
    # phi = list(zip(Q.max(axis=0) // va_eps, Q.argmax(axis=0)))
    phi = Q.argmax(axis=0)
    # phi = Q.max(axis=0) // va_eps
    va_states = list(set(phi))
    num_x = len(va_states)
    # use ANY STATIONARY VA-STATE BASED explorative policy to get a stationary distribution on T
    #pi_x = np.random.random_sample([len(va_states), num_a]) + eps
    # pi_x = np.random.beta(0.5,0.5,[len(va_states), num_a]) + eps
    # pi_x = pi_x/pi_x.sum(axis=1,keepdims=True)
    # pi_behavior = np.array([pi_x[va_states.index(x)] for x in phi])
    #pi_behavior = np.random.random_sample([num_s,num_a]) + eps
    pi_behavior = np.random.beta(0.5,0.5,[num_s,num_a]) + eps
    pi_behavior = pi_behavior/pi_behavior.sum(axis=1,keepdims=True)
    d = mdps.stationary_dist(T, pi_behavior)
    B = [d for a in range(num_a)]
    # use d and phi to get a surrogate MDP (T_mdp,R_mdp)
    print("Creating the surrogate MDP.")
    T_mdp,R_mdp = mdps.surrogate_mdp(T, R, phi, B, va_states)
    # get the optimal policy on this surrogate MDP using VI
    print("Value iteration on the surrogate MDP.")
    Q_mdp = algos.tabular_vi(T_mdp,R_mdp,g)
    pi_mdp = mdps.greedy_policy(Q_mdp)
    # use this optimal policy to get values of (T,R)
    pi_via_mdp = np.array([pi_mdp[va_states.index(x)] for x in phi])
    if np.linalg.norm(pi-pi_via_mdp) > 0:
        print("The surrogate MDP has FAILED to provide the optimal policy.")
        print("Value iteration on the original process using the uplifted policy.")
        Q_via_mdp = algos.tabular_vi(T, R, g, pi_via_mdp)
        # compare these values with the normalized Q
        print("Q-value difference norm = ", np.linalg.norm(Q-Q_via_mdp))
        print("Q-value difference d-weighted norm = ", weighted_norm(Q-Q_via_mdp, d))
    else:
        print("The surrogate MDP has SUCCESSFULLY provided the optimal policy.")
    print("Number of ground states = ", num_s)
    print("Number of abstract states = ", num_x)

def weighted_norm(A,w):
    return np.sqrt((w*A*A).sum())

va_analysis()