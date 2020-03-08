import mdps
import algos

import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(description="value-action aggregation analysis")

parser.add_argument("--num_x", default=4, type=int, help="number aggregated states (maximum)")
parser.add_argument("--num_a", default=2, type=int, help="number of actions")
parser.add_argument("--num_s", default=8, type=int, help="number of ground states")
parser.add_argument("-r", action="store_true", help="randomize state and action space sizes")
parser.add_argument("--g", default=0.999, type=float, help="discount factor")
parser.add_argument("--tries", default=100000, type=int, help="maximum simulation steps")
parser.add_argument("--va_eps", default=1e-6, type=float, help="aggregation tolerance")
parser.add_argument("--eps", default=1e-7, type=float, help="simulation tolerance")
parser.add_argument("--steps", default=np.inf, type=int, help="value/policy iteration maximum steps")
parser.add_argument("--greedy_eps", default=0.0, type=float, help="epsilon-greedy choice")
parser.add_argument("-n", action="store_true", help="normalize rewards")
parser.add_argument("-s", action="store_true", help="state-based behavior policy")
parser.add_argument("-u", action="store_true", help="uniform random behavior policy")
parser.add_argument("-a", action="store_true", help="aggregate all optimal actions")
parser.add_argument("--file", required=True, type=str, help="output file")

args = parser.parse_args()

def surrogate_mdp_analysis(num_a, num_s, num_x, args):
    # generate a random mdp
    T,R,_,rmin,rmax = mdps.random_mdp(num_a, num_s, args) 
    # generate a random phi
    phi = np.random.choice(range(num_x), num_s)
    # unique states
    va_states = range(num_x)
    # get a behavior policy to generate B
    # if state-based behavior
    if args.s:
        # use ANY STATIONARY VA-STATE BASED explorative policy to get a stationary distribution on T
        pi_x = mdps.random_policy(num_a, num_x, args)
        pi_behavior = np.array([pi_x[va_states.index(x)] for x in phi])
    else:
        # use ANY explorative policy to get a stationary distribution on T
        pi_behavior = mdps.random_policy(num_a, num_s, args)
    # pi_behavior = mdps.fixed_policy(num_a,num_s)
    d = mdps.stationary_dist_eigenvalue(T, args, pi_behavior)
    # build B
    B = pi_behavior * d[:, np.newaxis]
    B = B.T
    # get a surrogate MDP and its optimal policy
    # use B and phi to get a surrogate MDP (T_mdp,R_mdp)
    T_mdp,R_mdp = mdps.surrogate_mdp(T, R, phi, B, va_states)
    # get the optimal policy on this surrogate MDP using VI
    Q_mdp, _ = algos.tabular_vi(T_mdp, R_mdp, args)
    pi_mdp = mdps.greedy_policy(Q_mdp, args)
    # use the surrogate policy in the original mdp
    # use this optimal policy to get values of (T,R)
    pi_via_mdp = np.array([pi_mdp[va_states.index(x)] for x in phi])
    Q_via_mdp, V_via_mdp = algos.tabular_vi(T, R, args, pi_via_mdp)
    # evaluate the conjecture
    print((Q_via_mdp*B).sum() - Q_mdp)

surrogate_mdp_analysis(1, 3, 1, args)