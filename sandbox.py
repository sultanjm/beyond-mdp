import mdps
import algos
import numpy as np
import matplotlib.pyplot as plt

import pickle

def sandbox():
    pi1 = np.array([[1.0, 0.0]])
    pi2 = np.array([[0.5,0.5]])
    print(mdps.same_optimal_policies(pi1,pi2))
    print(mdps.same_optimal_policies(pi2,pi1))

sandbox()