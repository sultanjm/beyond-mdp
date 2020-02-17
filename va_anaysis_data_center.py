import mdps
import algos
import numpy as np
import matplotlib.pyplot as plt

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
parser.add_argument("--file", type=str, help="output file")

args = parser.parse_args()

def pickle_data_analysis_center(args):
    data = loadall("data-sets/va_cx_nrs_eps_5.pickle")
    msize = 3
    plt.figure(1)
    plt.subplot(221)
    v_plt = plt.gca()
    plt.subplot(222)
    v_norm_plt = plt.gca()
    plt.subplot(223)
    q_plt = plt.gca()
    plt.subplot(224)
    q_norm_plt = plt.gca()
    # plt.subplot(325)
    # v_log_plt = plt.gca()
    # plt.subplot(326)
    # q_log_plt = plt.gca()
    # v_log_plt.set_yscale('log')
    # q_log_plt.set_yscale('log')
    # v_log_plt.set_xscale('log')
    # q_log_plt.set_xscale('log')
    idx_d = 0
    for d in data:
        # if idx_d > 2000:
        #     break
        num_a, num_s, _ = d['T'].shape
        q_y = np.abs(d['Q'] - d['Q_via_mdp'])
        q_x = d['B']
        idx = d['Q'].argmax(axis=0)
        v_y = np.abs(d['Q'].max(axis=0) - d['Q_via_mdp'].max(axis=0))
        v_x = []
        s = 0
        for a in idx:
            v_x = np.hstack((v_x, d['B'][a][s])) if np.size(v_x) else d['B'][a][s]
            s += 1
        # v_x = mdps.stationary_dist_eigenvalue(d['T'], args, d['pi_via_mdp'])
        v_plt.scatter(v_x, v_y, s=msize)
        # v_log_plt.scatter(v_x, v_y, s=2)
        q_plt.scatter(q_x, q_y, s=msize)
        # q_log_plt.scatter(q_x, q_y, s=2)
        q_norm_plt.scatter(num_s + num_a, q_y.max()-q_y.min(), s=msize)
        v_norm_plt.scatter(num_s + num_a, v_y.max()-v_y.min(), s=msize)
        idx_d += 1
    # h_x = np.linspace(0,1,100)
    # h_y = f(h_x)
    # v_plt.plot(h_x, h_y)
    plt.show()

def f(x):
    sigma = 0.35
    A = 30
    return  100 * A * np.sqrt(sigma**2/np.pi) * np.exp(-x**2/sigma**2) + 1e-8
def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def weighted_norm(A,w):
    return np.sqrt((w*A*A).sum())

pickle_data_analysis_center(args)