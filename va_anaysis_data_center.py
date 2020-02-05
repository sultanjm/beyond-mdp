import mdps
import algos
import numpy as np
import matplotlib.pyplot as plt

import pickle

def pickle_data_analysis_center():
    data = loadall("va_counter_examples.pickle")
    plt.figure(1)
    plt.subplot(221)
    v_plt = plt.gca()
    plt.subplot(222)
    v_log_plt = plt.gca()
    plt.subplot(223)
    q_plt = plt.gca()
    plt.subplot(224)
    q_log_plt = plt.gca()
    v_log_plt.set_yscale('log')
    q_log_plt.set_yscale('log')
    # v_log_plt.set_xscale('log')
    # q_log_plt.set_xscale('log')
    for d in data:
        q_y = np.abs(d['Q'] - d['Q_via_mdp'])
        q_x = d['B']
        idx = d['Q'].argmax(axis=0)
        v_y = np.abs(d['Q'].max(axis=0) - d['Q_via_mdp'].max(axis=0))
        v_x = []
        s = 0
        for a in idx:
            v_x = np.hstack((v_x, d['B'][a][s])) if np.size(v_x) else d['B'][a][s]
            s += 1
        v_plt.scatter(v_x, v_y)
        v_log_plt.scatter(v_x, v_y)
        q_plt.scatter(q_x, q_y)
        q_log_plt.scatter(q_x, q_y)

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

pickle_data_analysis_center()