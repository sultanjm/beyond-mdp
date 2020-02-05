import mdps
import algos
import numpy as np
import matplotlib.pyplot as plt

import pickle

def converter():
    data = loadall("va_counter_examples.pickle")
    for d in data:
        rmin = d['rmin']
        rmax = d['rmax']
        R = (d['R'] - rmin)/(rmax-rmin)
        g = 0.999 # I should've saved it in the simulation!
        va_eps = 1e-6 # same
        T = d['T']
        Q = (d['Q'] - (rmin)/(1-g))/(rmax-rmin)
        B = d['B']
        pi = d['pi']
        phi = list(zip(Q.max(axis=0) // va_eps, Q.argmax(axis=0)))
        va_states = list(set(phi))
        T_mdp,R_mdp = mdps.surrogate_mdp(T, R, phi, B, va_states)
        Q_mdp = algos.tabular_vi(T_mdp,R_mdp,g)
        pi_mdp = mdps.greedy_policy_all(Q_mdp)
        pi_via_mdp = np.array([pi_mdp[va_states.index(x)] for x in phi])
        Q_via_mdp = Q
        if not mdps.same_optimal_policies(pi,pi_via_mdp):
            Q_via_mdp = algos.tabular_vi(T, R, g, pi_via_mdp)
            rtrn = dict()
            rtrn['success'] = 
            rtrn['T'] = T; rtrn['R'] = R; rtrn['rmin'] = rmin; rtrn['rmax'] = rmax
            rtrn['pi'] = pi; rtrn['pi_via_mdp'] = pi_via_mdp
            rtrn['Q'] = Q; rtrn['Q_via_mdp'] = Q_via_mdp
            rtrn['d'] = d; rtrn['Q_norm'] = np.linalg.norm(Q-Q_via_mdp); rtrn['B'] = B
            rtrn['va_eps'] = va_eps; rtrn['g'] = g
            rtrn['normalized_rewards'] = True
            with open('va_counter_examples_converted.pickle', 'ab+') as f:
                pickle.dump(rtrn, f)

def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

converter()