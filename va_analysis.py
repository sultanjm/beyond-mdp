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
parser.add_argument("--greedy_eps", default=1e-3, type=float, help="epsilon-greedy choice")
parser.add_argument("-n", action="store_true", help="normalize rewards")
parser.add_argument("-s", action="store_true", help="state-based behavior policy")
parser.add_argument("-u", action="store_true", help="uniform random behavior policy")
parser.add_argument("-a", action="store_true", help="aggregate all optimal actions")
parser.add_argument("--file", required=True, type=str, help="output file")

args = parser.parse_args()

def va_simulation(num_a, num_s, num_x, args):
    T,R,Q,rmin,rmax = mdps.random_va_mdp(num_a, num_s, num_x, args)
    # normalize rewards, so normalize Q with rmin and rmax
    if args.n:
        Q = (Q - rmin/(1-args.g))/(rmax-rmin)
    # get the (all) optimal policy
    pi = mdps.greedy_policy_all(Q, args)
    # do VA abstraction of T based on Q
    phi = list(mdps.va_states(Q, args))
    # unique va-states
    va_states = list(set(phi))
    # true number of abstract states
    num_x = len(va_states)
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
    # use B and phi to get a surrogate MDP (T_mdp,R_mdp)
    T_mdp,R_mdp = mdps.surrogate_mdp(T, R, phi, B, va_states)
    # get the optimal policy on this surrogate MDP using VI
    Q_mdp = algos.tabular_vi(T_mdp, R_mdp, args)
    #pi_mdp = mdps.greedy_policy(Q_mdp)
    pi_mdp = mdps.greedy_policy_all(Q_mdp, args)
    # use this optimal policy to get values of (T,R)
    pi_via_mdp = np.array([pi_mdp[va_states.index(x)] for x in phi])
    # check if surrogate MDP provides an optimal policy
    Q_via_mdp = Q
    success = True
    if not mdps.same_optimal_policies(pi, pi_via_mdp):
        Q_via_mdp = algos.tabular_vi(T, R, args, pi_via_mdp)
        success = False
    result_var = ['success', 
                  'T', 'R', 'rmin', 'rmax',
                  'pi', 'pi_via_mdp', 'pi_behavior',
                  'Q', 'Q_via_mdp', 'd', 'B', 
                  'args']
    result = dict()
    for i in result_var:
        result[i] = locals()[i]
    return result

def va_analysis(args):
    num_found = 0
    for t in range(args.tries):
        num_a = args.num_a + np.random.choice(range(2)) if args.r else args.num_a
        num_s = args.num_s + np.random.choice(range(4)) if args.r else args.num_s
        num_x = args.num_x + np.random.choice(range(2)) if args.r else args.num_x
        result = va_simulation(num_a, num_s, num_x, args)
        # we have found a counter example
        if result['success'] == False:
            num_found += 1
            with open(args.file, 'ab+') as f:
                pickle.dump(result, f)
        print("{}/{} Tries with {} counter-example(s) found.".format(t+1, args.tries, num_found), end='\r')

va_analysis(args)