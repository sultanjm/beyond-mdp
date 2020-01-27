import mdps
import algos

import numpy as np
import pickle

def va_simulation(num_s, num_a, num_x, g, va_eps, eps):
    T,R,Q,rmin,rmax = mdps.random_va_mdp(num_a,num_s,num_x,va_eps,g)
    # normalize Q with rmin and rmax
    # Q = (Q - rmin/(1-g))/(rmax-rmin)
    # get the optimal policy
    # pi = mdps.greedy_policy(Q)
    pi = mdps.greedy_policy_all(Q)
    # do VA abstraction of T based on Q
    phi = list(zip(Q.max(axis=0) // va_eps, Q.argmax(axis=0)))
    # phi = Q.argmax(axis=0)
    # phi = Q.max(axis=0) // va_eps
    va_states = list(set(phi))
    num_x = len(va_states)
    # use ANY explorative policy to get a stationary distribution on T
    # pi_behavior = mdps.random_policy(num_a,num_s)
    pi_behavior = mdps.fixed_policy(num_a,num_s)
    # d = mdps.stationary_dist(T, pi_behavior)
    d = mdps.stationary_dist_cesaro(T, pi_behavior)
    B = [d for a in range(num_a)]
    # use d and phi to get a surrogate MDP (T_mdp,R_mdp)
    # print("Creating the surrogate MDP.")
    T_mdp,R_mdp = mdps.surrogate_mdp(T, R, phi, B, va_states)
    # get the optimal policy on this surrogate MDP using VI
    # print("Value iteration on the surrogate MDP.")
    Q_mdp = algos.tabular_vi(T_mdp,R_mdp,g)
    #pi_mdp = mdps.greedy_policy(Q_mdp)
    pi_mdp = mdps.greedy_policy_all(Q_mdp)
    # use this optimal policy to get values of (T,R)
    pi_via_mdp = np.array([pi_mdp[va_states.index(x)] for x in phi])
    # check if surrogate MDP provides an optimal poilicy
    Q_via_mdp = Q
    success = True
    if not mdps.same_optimal_policies(pi,pi_via_mdp):
        Q_via_mdp = algos.tabular_vi(T, R, g, pi_via_mdp)
        # compare these values with the normalized Q
        success = False
    rtrn = dict()
    rtrn['success'] = success
    rtrn['T'] = T; rtrn['R'] = R; rtrn['rmin'] = rmin; rtrn['rmax'] = rmax
    rtrn['pi'] = pi; rtrn['pi_via_mdp'] = pi_via_mdp
    rtrn['Q'] = Q; rtrn['Q_via_mdp'] = Q_via_mdp
    rtrn['d'] = d; rtrn['Q_norm'] = np.linalg.norm(Q-Q_via_mdp); rtrn['Q_d_norm'] = weighted_norm(Q-Q_via_mdp, d); rtrn['B'] = B
    return rtrn

def weighted_norm(A,w):
    return np.sqrt((w*A*A).sum())

def va_analysis(tries=100, num_s=16, num_a=4, num_x=2, g=0.999, va_eps=1e-6, eps=1e-9):
    with open('va_counter_examples.pickle', 'ab+') as f:
        num_found = 0
        for t in range(tries):
            result = va_simulation(num_s, num_a, num_x, g, va_eps, eps)
            # we have found a counter example
            if result['success'] == False:
                num_found += 1
                pickle.dump(result, f)
            print("{}/{} Tries with {} counter-example(s) found.".format(t,tries,num_found), end='\r')
va_analysis()